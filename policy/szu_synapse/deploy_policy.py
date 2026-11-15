"""RoboSynChallenge entry points; task observations only, no expert state."""
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from runtime import load_policy

def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

def encode_observation(obs, prompt):
    state = _array(obs['robot']['qpos'])
    if state.ndim == 2:
        if state.shape[0] != 1:
            raise ValueError('This policy supports one environment at a time')
        state = state[0]
    if state.shape == (32,):
        state = state[np.r_[0:7, 16:23]]
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Expected 14 active joints/grippers, or the documented 32D recording layout')
    result = {'observation/state': state.astype(np.float32), 'prompt': prompt}
    for camera, key in [('cam_high', 'image'), ('cam_left_wrist', 'left_wrist_image'),
                        ('cam_right_wrist', 'right_wrist_image')]:
        image = _array(obs['sensor'][camera]['color'])
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError('Expected a single camera image')
            image = image[0]
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError('Expected HWC RGB/RGBA camera image')
        image = image[..., :3]
        if image.dtype != np.uint8:
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError('Floating camera images must use the documented 0..1 range')
            image = (image * 255).astype(np.uint8)
        result['observation/' + key] = image
    return result

class TaskPolicy:
    def __init__(self, args):
        self.task = args['task_name']
        bindings = json.loads((ROOT / 'task_bindings.json').read_text())
        self.binding = bindings[self.task]
        checkpoint = args.get('checkpoint_path')
        if not checkpoint:
            checkpoint = ROOT / 'checkpoints' / self.task
        self.policy = load_policy(checkpoint, self.binding, platform=args.get('jax_platform'))
        self.execute_steps = int(args.get('execute_steps', 10))
        if not 1 <= self.execute_steps <= 50:
            raise ValueError('execute_steps must be in 1..50')
        self.prompt = self.binding['prompt']

    def infer(self, observation, *, noise=None):
        result = self.policy.infer(encode_observation(observation, self.prompt), noise=noise)
        actions = np.asarray(result['actions'], dtype=np.float32)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError('Invalid policy action chunk')
        # Both simulator and real interface use absolute normalized gripper commands.
        actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0, 1)
        return actions

def get_model(usr_args):
    return TaskPolicy(usr_args)

def eval(env, model, observation):
    import torch
    started = time.perf_counter()
    actions = model.infer(observation)[:model.execute_steps]
    # np.asarray in infer waits for device execution, so this includes inference.
    inference_times = [time.perf_counter() - started]
    action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if action_dim != 14:
        raise ValueError(f'Expected the official 14D action interface, got {action_dim}')
    info = None
    truncated = False
    for action in actions:
        tensor = torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
        observation, reward, terminated, truncated, info = env.step(tensor)
        if bool(_array(terminated).any()) or bool(_array(truncated).any()):
            break
        if env.get_wrapper_attr('is_task_success')():
            break
    return observation, info, truncated, inference_times

def reset_model(model):
    # Receding-horizon inference has no observation/action queue between calls.
    pass
