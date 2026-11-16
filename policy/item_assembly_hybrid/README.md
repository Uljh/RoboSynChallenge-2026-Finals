# Item assembly mixed VLA + frozen visual servo

This package relocates the tested native simulation controller and adds an
explicitly optional static-kinematics adapter. The geometry network, edge fit,
temporal filter and handoff thresholds are unchanged. The static adapter always
uses the original bounded local-IK fallback instead of native IK. That change
has an offline interface smoke test, not the native controller's 10/12 score.
There are no H800 import paths or required training experiments.

The selected mixed VLA checkpoint is step 1500, chosen by the minimum fixed
REAL validation loss among steps 1000/1500/2000. Its parameters and `assets/`
must remain together. Optional hybrid modes also require
`geometry_root/full_fit_1000.pt`:

```
SHA256 e3aa4bd4ac3aea54e5c39b3ace30829565767a52f0a671a2d958e8aff6e3088d
size   3351176 bytes
```

The geometry model is supplied under `item_assembly/geometry/` in the
[weight repository](https://huggingface.co/lige65456/robosyn-finals-2026-pi05).
The unified task downloader verifies it with the checkpoint manifest. Set
`geometry_root` to the downloaded task's `geometry/` directory. The two small
nominal meshes and FK chain are in `assets/`.
Use the clean `policy/szu_synapse/runtime.py` loader and the tested versions in
`deployment_versions.json`. Hybrid inference additionally needs trimesh, scipy,
torch and OpenCV. The standalone `api.load_vla` remains a research-environment
loader that imports public training config and its dependencies (including
tyro/LeRobot); do not use it in the clean submission runtime.

## Integration

Recommended integration with the unified clean runtime:

```python
from policy.szu_synapse.runtime import load_policy
from policy.item_assembly_hybrid import create_hybrid, KinematicGate

vla = load_policy(checkpoint_path, selected_binding)
hybrid = create_hybrid(
    vla, checkpoint_path, geometry_root,
    log_directory=episode_log_directory, profile="frozen_sim",
)
# On each reset:
hybrid.reset()
# Build KinematicGate with the observed initial robot state, the existing robot
# FK backend, and assets/handoff_contract.json, just as in the frozen collector.
# Keep `previous` in the same 14-D joint/ratiometric-gripper convention.
actions = hybrid.joint_actions(
    observation, control_step, robot_environment, tensor_to_numpy,
    gate, previous, joint_limits, episode_log_directory,
)
```

Canonical observation keys are `observation/image`,
`observation/left_wrist_image`, `observation/right_wrist_image`,
`observation/state`, and `prompt`. RGB is uint8 HWC 480x640; the state is the
original 14-D joint/gripper observation. The VLA uses its checkpoint's norm.
The handoff gate deliberately retains the original bounds in
`assets/stage_contract.json`; replacing those with mixed-data normalization would
change the controller.

`joint_actions` returns 50x14 base actions (execute at most 10), or 1x14 precise
joint targets (execute one before reading fresh observation). It uses the
original native `robot.compute_fk/compute_ik` backend, with bounded local IK
fallback. `KinematicGate` requires robot finger-link transforms; it does not read
the task objects. Native simulation encoding multiplies gripper ratios by 0.05;
do not apply that conversion twice or silently reuse it for another backend.

**`request("precision")` and the legacy `infer()` stage interface return geometry
packets, NOT robot actions.** The first nine values are endpoint delta and two
axes. Always feed those through `LearnedServo`, as `joint_actions` does. Sending a
geometry packet directly to the robot is incorrect.

`deploy_policy.py` additionally exports official `get_model / eval / reset_model`
entry points. Supply `checkpoint_path`, exact selected `binding` and `prompt`;
or inject an already loaded canonical `base_policy`. Default `mode='vla'` uses
the clean runtime and normalized gripper commands. Optional modes are
`frozen_sim_native` (explicit robot backend) and `frozen_sim_static` (observed
joints plus packaged kinematics); both also require `geometry_root` and explicit
14x2 `joint_limits`. Never copy a simulator fixture's limits to hardware.
`action_gripper_convention='sim_physical_005'` is an explicit simulation-only
option. The wrapper executes ten coarse actions or one precision action and
resets its stage, geometry filters and previous command between episodes.
No profile is a certified real hybrid or an automatic official Submit action.

## Verified scope and real-robot gap

Relocated geometry produced exactly the same outputs as the original on four
recorded RGB/state inputs and after reset, using the real geometry checkpoint.
That original CPU geometry comparison is separate from selected-model checks.
The selected mixed VLA has now passed actual clean-runtime loading on 194,
36 recorded-observation probes with repeatability, and an offline interface
check with the geometry controller. See `selected_runtime_smoke.json`.
The static full geometry/IK action path passed a CPU interface smoke using the
actual geometry model and recorded episode640/frame290 RGB: natural gate entry,
one finite 14D joint target, maximum left-arm update0.003206rad, right arm held,
reset reproducibility and explicit gripper conversion. Coarse actions were a
recorded-action fixture; no new VLA or closed-loop physics was tested. See
`controller_smoke.json` for the precise scope. Static midpoint checks measure
self-consistency, not independent calibration accuracy.

The only available profile is `frozen_sim`. The old 10/12 result belongs to the
old VLA plus this simulation visual controller, not the new checkpoint, pure
VLA, or a real-robot test.

Official November 15 calibration packages now provide raw AprilTag observations
and intrinsics. All 58 records are high-camera eye-to-hand captures; platforms 1
and 2 also include wrist images/intrinsics, while platform 3 has only high-camera
images. No ready solved wrist-to-link6 transform or tube CAD is included in
those archives. See `official_calibration_audit.json` for sources and hashes.

This frozen implementation uses the nominal simulation `guijiao_standard`
meshes, one fixed intrinsics vector for both wrists, fixed camera extrinsics and
simulation robot mounting geometry. They have **not** been validated against the
real centrifuge-tube object, both wrists, or every evaluation platform. Raw
calibration observations may support other fitting approaches, but the default
AprilTag detector found tags in58/58 high-camera images and0/40 wrist images.
Platform3 has no wrist samples. These records have not yielded reliable wrist
extrinsics for this controller. See `real_calibration_feasibility.json`.
`profile="real"` is therefore not available. Do not
describe the module as real-robot-ready or replace the frozen calibration
silently. Pure VLA loading is independent of these geometry prerequisites.
