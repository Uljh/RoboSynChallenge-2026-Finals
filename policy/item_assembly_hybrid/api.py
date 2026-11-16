"""Explicit paths, no implicit training launch or host-specific imports."""
from pathlib import Path
import hashlib,json,dataclasses
import numpy as np
from .controller import FrozenHybridAdapter
GEOMETRY_SHA256='e3aa4bd4ac3aea54e5c39b3ace30829565767a52f0a671a2d958e8aff6e3088d'
ASSETS=Path(__file__).resolve().parent/'assets'
def verify_geometry_checkpoint(root):
 p=Path(root)/'full_fit_1000.pt'
 with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 if actual!=GEOMETRY_SHA256:raise ValueError('Geometry model hash mismatch')
 return p
def load_vla(checkpoint,default_prompt=None,device=None):
 import jax
 import jax.numpy as jnp
 from openpi.training import config
 from openpi.models import model as model_lib
 from openpi.policies.policy import Policy
 from openpi import transforms
 from openpi.shared import normalize
 cp=Path(checkpoint).resolve();norms=list((cp/'assets').rglob('norm_stats.json'))
 if len(norms)!=1:raise ValueError('Expected exactly one checkpoint normalization asset')
 asset_id=str(norms[0].parent.relative_to(cp/'assets'))
 # Use the public released config, not a training-host-only registration.
 # V2.1 differs in training repack only; inference starts from the canonical keys.
 cfg=dataclasses.replace(config.get_config('pi05_base_robosynchallenge_full'),fsdp_devices=1,
  data=config.LeRobotEmbodiChainDataConfig(repo_id=asset_id,assets=config.AssetsConfig(assets_dir=str(cp/'assets'),asset_id=asset_id),base_config=config.DataConfig(prompt_from_task=True),extra_delta_transform=True))
 target=device if device is not None else jax.devices()[0]
 # Public policy_config lacks restore_sharding; preserve explicit single-device
 # restore using the public model API and its identical transform sequence.
 with jax.default_device(target):
  model=cfg.model.load(model_lib.restore_params(cp/'params',dtype=jnp.bfloat16,sharding=jax.sharding.SingleDeviceSharding(target)))
  dc=cfg.data.create(cfg.assets_dirs,cfg.model);norm=normalize.load(norms[0].parent)
  return Policy(model,transforms=[transforms.InjectDefaultPrompt(default_prompt),*dc.data_transforms.inputs,transforms.Normalize(norm,use_quantiles=dc.use_quantile_norm),*dc.model_transforms.inputs],output_transforms=[*dc.model_transforms.outputs,transforms.Unnormalize(norm,use_quantiles=dc.use_quantile_norm),*dc.data_transforms.outputs],sample_kwargs={'num_steps':10},metadata=cfg.policy_metadata)
def create_hybrid(base_policy,checkpoint,geometry_root,log_directory=None,profile='frozen_sim'):
 # There is no certified real profile yet. Do not silently relabel simulation CAD/calibration.
 if profile!='frozen_sim':raise ValueError('Only frozen_sim is available; real wrist extrinsics and object geometry have not been validated')
 from openpi.shared import normalize
 from .learned_geometry import LearnedGeometry
 verify_geometry_checkpoint(geometry_root)
 norms=list((Path(checkpoint)/'assets').rglob('norm_stats.json'))
 if len(norms)!=1:raise ValueError('Expected one norm asset')
 norm=normalize.load(norms[0].parent);contract=json.loads((ASSETS/'stage_contract.json').read_text())
 contract['norm_hashes']['base']=hashlib.sha256(norms[0].read_bytes()).hexdigest()
 contract['encoder_state_low']=np.asarray(norm['state'].q01).tolist();contract['encoder_state_high']=np.asarray(norm['state'].q99).tolist()
 geometry=LearnedGeometry(geometry_root,log_directory=log_directory)
 return FrozenHybridAdapter(base_policy,geometry,contract)
