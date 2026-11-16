"""Official entry points, with pure VLA default and explicit optional sim hybrid.

Pass base_policy from the clean submission runtime to avoid importing training
configuration. frozen_sim_static uses observed joints and bundled robot geometry
only; it has NOT reproduced the native-controller 10/12 result or passed real
robot calibration. No precision feature packet is sent to env.step.
"""
from pathlib import Path
import json
import tempfile
import time
import numpy as np
from .static_backend import StaticRobotBackend, as_array


def encode_observation(obs, prompt):
    state = as_array(obs['robot']['qpos'])
    if state.ndim == 2 and state.shape[0] == 1:
        state = state[0]
    if state.shape == (32,):
        state = state[np.r_[0:7, 16:23]]
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Expected finite 14D state or documented 32D recording layout')
    out = {'observation/state': state.astype(np.float32), 'prompt': prompt}
    for camera, key in [('cam_high', 'image'), ('cam_left_wrist', 'left_wrist_image'),
                        ('cam_right_wrist', 'right_wrist_image')]:
        image = as_array(obs['sensor'][camera]['color'])
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError('Expected single HWC RGB/RGBA image')
        image = image[..., :3]
        if image.dtype != np.uint8:
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError('Floating images must be in 0..1')
            image = (image * 255).astype(np.uint8)
        out['observation/' + key] = image
    return out


class DeploymentPolicy:
    def __init__(self, args):
        self.mode = args.get('mode', 'vla')
        if self.mode not in ('vla', 'frozen_sim_static', 'frozen_sim_native'):
            raise ValueError('No certified real-hybrid mode is available')
        self.prompt = args['prompt']
        self.base = args.get('base_policy')
        if self.base is None:
            # The submission loader bundles its tokenizer and imports no training
            # registry / LeRobot / tyro. The legacy api.load_vla is research-only.
            from policy.szu_synapse.runtime import load_policy
            binding = args['binding']
            if binding['prompt'] != self.prompt:
                raise ValueError('Prompt must match the selected inference binding')
            self.base = load_policy(args['checkpoint_path'], binding, platform=args.get('jax_platform'))
        self.convention = args.get('action_gripper_convention', 'normalized')
        if self.convention not in ('normalized', 'sim_physical_005'):
            raise ValueError('Unknown action gripper convention')
        if self.convention == 'sim_physical_005' and self.mode == 'vla':
            raise ValueError('Physical .05 conversion is only for an explicitly selected sim profile')
        self.hybrid = None
        self.backend = args.get('robot_backend')
        self.output = Path(args.get('log_directory') or tempfile.mkdtemp(prefix='assembly_hybrid_'))
        self.output.mkdir(parents=True, exist_ok=True)
        if self.mode != 'vla':
            from .api import create_hybrid
            self.limits = np.asarray(args['joint_limits'], dtype=np.float64)
            if self.limits.shape != (14, 2) or not np.isfinite(self.limits).all() or np.any(self.limits[:, 0] >= self.limits[:, 1]):
                raise ValueError('Supply the deployment robot 14x2 limits explicitly')
            if self.mode == 'frozen_sim_native' and self.backend is None:
                raise ValueError('Explicit native robot backend required')
            self.hybrid = create_hybrid(self.base, args['checkpoint_path'], args['geometry_root'],
                                        log_directory=self.output, profile='frozen_sim')
        self.reset()

    def reset(self):
        self.control = 0
        self.previous = None
        self.gate = None
        if self.hybrid is not None:
            self.hybrid.reset()
        if hasattr(self.base, 'reset'):
            self.base.reset()

    def infer(self, observation):
        obs = encode_observation(observation, self.prompt)
        state = obs['observation/state']
        if self.hybrid is None:
            batch = np.asarray(self.base.infer(obs)['actions'], dtype=np.float32)
            if batch.shape != (50, 14):
                raise ValueError('Expected 50x14 VLA actions')
            batch = batch[:10].copy()
        else:
            from .kinematic_gate import KinematicGate
            if self.mode == 'frozen_sim_static':
                if self.gate is None:
                    self.backend = StaticRobotBackend(state)
                else:
                    self.backend.update(state)
            if self.gate is None:
                contract = json.loads((Path(__file__).parent / 'assets/handoff_contract.json').read_text())
                self.gate = KinematicGate(self.backend, state, as_array, contract, self.output)
                self.previous = state.copy()
            batch = self.hybrid.joint_actions(obs, self.control, self.backend, as_array, self.gate,
                                              self.previous, self.limits, self.output)
            batch = np.asarray(batch[:1 if self.hybrid.phase == 'precision' else 10], dtype=np.float32).copy()
        if not np.isfinite(batch).all():
            raise ValueError('Non-finite action')
        batch[:, [6, 13]] = np.clip(batch[:, [6, 13]], 0, 1)
        return batch

    def commit(self, canonical_action):
        self.previous = np.asarray(canonical_action, dtype=np.float32).copy()
        self.control += 1


def get_model(usr_args):
    return DeploymentPolicy(usr_args)


def eval(env, model, observation):
    import torch
    started = time.perf_counter()
    actions = model.infer(observation)
    inference_times = [time.perf_counter() - started]
    if int(np.prod(env.unwrapped.single_action_space.shape)) != 14:
        raise ValueError('Expected official 14D action interface')
    info, truncated = None, False
    for canonical in actions:
        command = canonical.copy()
        if model.convention == 'sim_physical_005':
            command[[6, 13]] *= .05
        observation, reward, terminated, truncated, info = env.step(
            torch.as_tensor(command, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0))
        model.commit(canonical)
        if bool(as_array(terminated).any()) or bool(as_array(truncated).any()):
            break
        if env.get_wrapper_attr('is_task_success')():
            break
    return observation, info, truncated, inference_times


def reset_model(model):
    model.reset()
