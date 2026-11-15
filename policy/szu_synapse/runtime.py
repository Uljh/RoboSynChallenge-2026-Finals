"""Portable pi0.5 inference with the exact trained observation/action transforms.

The implementation uses the OpenPI sources shipped under policy/pi05, but does
not import any experiment registry, training dataset, or machine-specific path.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

POLICY_ROOT = Path(__file__).resolve().parent
OPENPI_ROOT = POLICY_ROOT.parent / 'pi05'
for path in [OPENPI_ROOT / 'src', OPENPI_ROOT / 'packages/openpi-client/src']:
    sys.path.insert(0, str(path))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

def load_policy(checkpoint, binding, *, platform=None):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.policy import Policy
    from openpi.policies.libero_policy import EmbodiChainInputs, EmbodiChainOutputs
    from openpi.shared import normalize
    from openpi import transforms as t
    import sentencepiece

    class BundledTokenizer(PaligemmaTokenizer):
        """Use the unchanged OpenPI tokenization algorithm with a bundled model."""
        def __init__(self):
            self._max_len = 200
            path = POLICY_ROOT / 'assets/paligemma_tokenizer.model'
            manifest = json.loads((path.parent / 'tokenizer_manifest.json').read_text())
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
                raise ValueError('Tokenizer checksum mismatch')
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=raw)

    checkpoint = Path(checkpoint).resolve()
    norm_path = checkpoint / binding['normalization_relative_path']
    if hashlib.sha256(norm_path.read_bytes()).hexdigest() != binding['normalization_sha256']:
        raise ValueError('Normalization does not match the selected checkpoint')
    norm = normalize.load(norm_path.parent)
    config = Pi0Config(pi05=True, action_dim=32, action_horizon=50, max_token_len=200)
    device = jax.devices(platform)[0] if platform else jax.devices()[0]
    params = model_lib.restore_params(checkpoint / 'params', dtype=jnp.bfloat16,
                                     sharding=jax.sharding.SingleDeviceSharding(device))
    model = config.load(params)
    delta_mask = t.make_bool_mask(6, -1, 6, -1)
    # Identical order to LeRobotEmbodiChainV21DataConfig + ModelTransformFactory.
    return Policy(
        model,
        transforms=[
            t.InjectDefaultPrompt(binding['prompt']),
            EmbodiChainInputs(model_type=config.model_type),
            t.DeltaActions(delta_mask),
            t.Normalize(norm, use_quantiles=True),
            t.InjectDefaultPrompt(None),
            t.ResizeImages(224, 224),
            t.TokenizePrompt(BundledTokenizer(), discrete_state_input=True),
            t.PadStatesAndActions(32),
        ],
        output_transforms=[
            t.Unnormalize(norm, use_quantiles=True),
            t.AbsoluteActions(delta_mask),
            EmbodiChainOutputs(),
        ],
        sample_kwargs={'num_steps': 10},
    )
