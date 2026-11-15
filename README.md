# SZU-VCC Synapse: RoboSynChallenge 2026 finals

Task-specific pi0.5 policies adapted from simulation to the released physical demonstrations.

This is a **policy-only overlay** based on official RoboSynChallenge commit
`45d9b9c4c78c66ef340a055697f3ad833ce567f4`. The official environment, task definitions,
and scoring code are supplied by the evaluator. Every runtime file in this
release is under `policy/`; no modifications to the environment or scoring are required.

The release is being assembled. Seven checkpoint selections are fixed; the
assembly, drawer and water adaptation runs are still being finalized. A weight
upload or interface check alone is not a completed submission.

- Code: <https://github.com/Uljh/RoboSynChallenge-2026-Finals>
- Weights: <https://huggingface.co/lige65456/robosyn-finals-2026-pi05>
- Policy entry: `policy/szu_synapse`
- Per-task model/normalization binding: `policy/szu_synapse/task_bindings.json`
- Upstream runtime provenance: `policy/szu_synapse/upstream.json`

## Installation

Use Python 3.11 and the official RoboSynChallenge/EmbodiChain evaluation
environment. Overlay this repository's `policy/` directory onto that checkout,
then install the inference requirements into the Python environment used by the
official evaluator:

```bash
git clone https://github.com/Uljh/RoboSynChallenge-2026-Finals.git synapse-release
cp -a synapse-release/policy/. /path/to/RoboSynChallenge/policy/
cd /path/to/RoboSynChallenge
python -m pip install -r policy/szu_synapse/requirements.txt
```

The supplied OpenPI source is imported directly from `policy/pi05/src` and
`policy/pi05/packages/openpi-client/src`; private experiment registries and
training datasets are not required. The tokenizer is bundled and verified by
SHA256, so inference does not require a Google Cloud download. JAX parameters are
restored onto a single visible device with preallocation disabled. A CUDA 12
environment and one GPU with sufficient free memory are required for practical
robot control; the CPU path is provided for installation and interface checks.
The historical inference environment used JAX 0.5.3, Flax 0.10.2,
Orbax 0.11.13 and Torch 2.7.1.

For a separate inference-only smoke environment (without the simulator):

```bash
python3.11 -m venv .venv-policy
.venv-policy/bin/python -m pip install -r policy/szu_synapse/requirements.txt
```

## Weights and task commands

Each task is a subdirectory of the same Hugging Face model repository. Download
the selected task and verify all files before using it:

```bash
python policy/szu_synapse/download_weights.py items_handover --revision main
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' python policy/szu_synapse/smoke_test.py \
  items_handover "$PWD/policy/szu_synapse/checkpoints/items_handover" --platform cpu
```

The smoke uses synthetic observations and checks a finite 50-by-14 action chunk.
It does not measure task success. Use a recorded immutable model commit from the
release manifest instead of `main` when reproducing a frozen release.

Run these commands from the official repository root after downloading the
corresponding task. Set `PYTHON_BIN` if the official environment's interpreter is
not the active `python`:

```bash
bash policy/szu_synapse/eval.sh click_bell "$PWD/policy/szu_synapse/checkpoints/click_bell" 0 --setting random
bash policy/szu_synapse/eval.sh drawer_open_place "$PWD/policy/szu_synapse/checkpoints/drawer_open_place" 0 --setting random
bash policy/szu_synapse/eval.sh handle_basket "$PWD/policy/szu_synapse/checkpoints/handle_basket" 0 --setting random
bash policy/szu_synapse/eval.sh item_assembly "$PWD/policy/szu_synapse/checkpoints/item_assembly" 0 --setting random
bash policy/szu_synapse/eval.sh items_handover "$PWD/policy/szu_synapse/checkpoints/items_handover" 0 --setting random
bash policy/szu_synapse/eval.sh manipulate_pipette "$PWD/policy/szu_synapse/checkpoints/manipulate_pipette" 0 --setting random
bash policy/szu_synapse/eval.sh mixer_operating "$PWD/policy/szu_synapse/checkpoints/mixer_operating" 0 --setting random
bash policy/szu_synapse/eval.sh sample_loading "$PWD/policy/szu_synapse/checkpoints/sample_loading" 0 --setting random
bash policy/szu_synapse/eval.sh table_rearrangement "$PWD/policy/szu_synapse/checkpoints/table_rearrangement" 0 --setting random
bash policy/szu_synapse/eval.sh water_pouring "$PWD/policy/szu_synapse/checkpoints/water_pouring" 0 --setting random
```

The three unfinished task bindings become usable only when their selected
checkpoint and paired normalization are included in the final release manifest.

## Runtime contract

`get_model(usr_args)` takes `task_name`, `checkpoint_path`, optional
`jax_platform` and `execute_steps` (default 10). `eval(env, model, observation)`
uses the official interface and returns observation, info, truncation and
inference durations. `reset_model(model)` discards no hidden task information;
ordinary inference keeps no action queue between calls.

Images are HWC RGB/RGBA arrays under `sensor.cam_high.color`,
`sensor.cam_left_wrist.color` and `sensor.cam_right_wrist.color`. A single
environment batch dimension is accepted. `robot.qpos` is the 14D active state;
the documented 32D recording layout is mapped using indices 0:7 and 16:23.
The output is 14 absolute targets: six left joints, left gripper, six right
joints, right gripper. Joint units are radians, grippers are ratios in [0,1].
Gripper outputs are clipped to that command range. No simulated finger-travel
multiplier is applied to the real interface. The model predicts 50 actions and
executes ten before observing again at the nominal 10Hz control rate.

No expert policy, task-object ground truth, or hidden simulator state is read to
construct the ordinary policy actions.

## Assembly-specific status

The selected mixed VLA will be deployable independently of the optional frozen
simulation visual-servo component. That component currently depends on
simulation tube geometry and wrist calibration. Its reported historical 10/12
result is a hybrid simulation result, not pure-VLA or physical success. It is not
enabled for real evaluation unless its geometry, calibration and outer control
interface have been explicitly validated. The final release will state which
mode is bound to the official task entry.

## Evidence and licenses

The seven existing selected checkpoints passed deterministic comparison against
their established loader: identical pre/post transforms and maximum absolute
generated-action difference 0.0 for the fixed CPU probe. A separate clean
environment and bundled-source model load/inference smoke also passed. These
checks establish packaging consistency, not physical success.

OpenPI code is distributed under its Apache-2.0 license in `policy/pi05/LICENSE`.
The fine-tuned model derivatives are subject to the supplied Gemma Terms of Use,
including Section 3.2 use restrictions, and the model repository's NOTICE.
No training datasets, optimizer state or account credentials are distributed.
