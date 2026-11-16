# SZU-VCC Synapse: RoboSynChallenge 2026 finals

Task-specific pi0.5 policies adapted from simulation to the released physical demonstrations.

This is a **policy-only overlay** based on official RoboSynChallenge commit
`45d9b9c4c78c66ef340a055697f3ad833ce567f4`. The official environment, task definitions,
and scoring code are supplied by the evaluator. Every runtime file in this
release is under `policy/`; no modifications to the environment or scoring are required.

The ten selected inference checkpoints, their paired normalization statistics,
and the technical report are included in this release. The weights have been
checked against source SHA256 manifests; all ten selected policies passed
actual loading and deterministic comparison with the established loader.
These packaging checks do not measure physical closed-loop task success.

- Code: <https://github.com/Uljh/RoboSynChallenge-2026-Finals>
- Weights: <https://huggingface.co/lige65456/robosyn-finals-2026-pi05>
- Policy entry: `policy/szu_synapse`
- Per-task model/normalization binding: `policy/szu_synapse/task_bindings.json`
- Upstream runtime provenance: `policy/szu_synapse/upstream.json`
- Frozen artifacts: `policy/szu_synapse/release_manifest.json`
- [Technical report](policy/szu_synapse/report/SZU_VCC_Synapse_Technical_Report.pdf)

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

All ten task bindings are finalized. Use the exact paired normalization in
each task directory. The manifest records immutable model revisions and every
published file count and download size.

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

## Assembly refinement

The default task binding loads the selected step-1500 mixed VLA through
`policy/szu_synapse`. An optional image-guided refinement module is provided
under `policy/item_assembly_hybrid`. It combines wrist-camera geometry,
temporal filtering, a stage handoff gate, and bounded kinematic corrections.
Its geometry model is under `item_assembly/geometry/` in the weight repository.
See the module README for configuration and integration.

## Artifacts and licenses

The release manifest records task bindings, model revisions, normalization
hashes, and the technical report. Per-task model manifests list file sizes and
SHA256 hashes for download verification.

OpenPI code is distributed under `policy/pi05/LICENSE` (Apache-2.0).
The model repository supplies the applicable Gemma Terms and NOTICE.

## Selected adaptation checkpoints

| Task | Step |
|---|---:|
| `click_bell` | 1500 |
| `drawer_open_place` | 3000 |
| `handle_basket` | 1500 |
| `item_assembly` | 1500 |
| `items_handover` | 1500 |
| `manipulate_pipette` | 2000 |
| `mixer_operating` | 4000 |
| `sample_loading` | 1500 |
| `table_rearrangement` | 1500 |
| `water_pouring` | 1500 |

Steps count adaptation updates after each selected simulation checkpoint.
Water step 1500 and sample-loading step 1500 are explicit user selections;
other task-specific selection rules are described in the report.
