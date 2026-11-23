"""RoboSynChallenge entry points; task observations only, no expert state."""
from pathlib import Path
import json
import sys
import time
import numpy as np

try:
    from .timing import EpisodeTiming
    from .chunk_execution import bind_chunk_executor as bind_chunk_executor, execute_chunk
except ImportError:
    from timing import EpisodeTiming
    from chunk_execution import bind_chunk_executor as bind_chunk_executor, execute_chunk

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
        started = time.perf_counter()
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
        self.load_seconds = time.perf_counter() - started
        self.timing = EpisodeTiming(self.task, self.load_seconds)
        self.last_inference_timing = {}

    def infer(self, observation, *, noise=None):
        started = time.perf_counter()
        encoded = encode_observation(observation, self.prompt)
        encoded_at = time.perf_counter()
        result = self.policy.infer(encoded, noise=noise)
        actions = np.asarray(result['actions'], dtype=np.float32)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError('Invalid policy action chunk')
        # Both simulator and real interface use absolute normalized gripper commands.
        actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0, 1)
        self.last_inference_timing = {
            'observation_encode_s': encoded_at - started,
            # CPU output materialization in Policy.infer waits for JAX sampling.
            'model_inference_s': time.perf_counter() - encoded_at,
            'sampling_s': result.get('policy_timing', {}).get('infer_ms', 0) / 1000,
        }
        return actions

def get_model(usr_args):
    return TaskPolicy(usr_args)

def predict_action_chunk(model, observation, *, noise=None):
    """Return the existing execution prefix, not all 50 predicted actions.

    One call performs one prediction from a measured observation. No command is
    issued here. The host may pass this prefix to its reviewed ordered executor.
    """
    actions = model.infer(observation, noise=noise) if noise is not None else model.infer(observation)
    prefix = np.array(actions[:model.execute_steps], dtype=np.float32, copy=True, order='C')
    prefix.setflags(write=False)
    return prefix


def eval(env, model, observation):
    executor = getattr(model, 'chunk_executor', None)
    if executor is None:
        return _eval_sequential(env, model, observation)
    timing = model.timing
    phase = 'inference'
    result = None
    try:
        if getattr(model, '_chunk_failed', False):
            phase = 'previous_chunk_failure'
            raise RuntimeError('Previous chunk failed; reconcile execution and bind a reviewed executor before continuing')
        if getattr(model, '_chunk_stopped', False):
            phase = 'episode_already_stopped'
            raise RuntimeError('Executor reported an episode stop; host must end/reset the environment before further actions')
        timing.event('phase_start', phase=phase)
        started = time.perf_counter()
        actions = predict_action_chunk(model, observation)
        inference_s = time.perf_counter() - started
        phase = 'chunk_execution_feedback'
        timing.event('phase_start', phase=phase, requested_actions=len(actions))
        timing.dispatch()
        started = time.perf_counter()
        result = execute_chunk(executor, model.task, actions)
        roundtrip_s = time.perf_counter() - started
        model._chunk_stopped = bool(result.success or result.terminated or result.truncated)
        # Validate a received observation without replacing it with commands.
        phase = 'chunk_response_validation'
        encode_observation(result.observation, model.prompt)
        timing.record_batch(inference_s, model.last_inference_timing, roundtrip_s, result)
        return result.observation, result.info, result.truncated, [inference_s]
    except Exception as exc:
        if phase in ('chunk_execution_feedback', 'chunk_response_validation'):
            model._chunk_failed = True
        timing.event('unrecoverable_error', phase=phase,
                     request_id=result.request_id if result is not None else getattr(exc, 'request_id', None),
                     completed_actions_this_call=int(result.executed_steps) if result is not None else getattr(exc, 'executed_steps', None),
                     error_type=type(exc).__name__, error=str(exc),
                     retry_performed=False, sequential_fallback_performed=False)
        raise


def _eval_sequential(env, model, observation):
    import torch
    timing = model.timing
    phase = 'inference'
    step_times = []
    feedback_times = []
    actions_issued = 0
    try:
        timing.event('phase_start', phase=phase)
        started = time.perf_counter()
        actions = predict_action_chunk(model, observation)
        inference_times = [time.perf_counter() - started]
        phase = 'action_preparation'
        started = time.perf_counter()
        action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
        if action_dim != 14:
            raise ValueError(f'Expected the official 14D action interface, got {action_dim}')
        # Stage the already predicted prefix with one device transfer. The public
        # simulator has only step(); do not invent a real-robot chunk RPC here.
        tensors = torch.tensor(actions, dtype=torch.float32,
                                  device=env.unwrapped.device).unsqueeze(1)
        if tensors.is_cuda:
            torch.cuda.synchronize(tensors.device)
        prepare_s = time.perf_counter() - started
        info = None
        truncated = False
        stopped = None
        timing.event('phase_start', phase='execution_feedback', requested_actions=len(tensors))
        for tensor in tensors:
            phase = 'env.step'
            started = time.perf_counter()
            # On an exception the action may already have reached the robot.
            # Never retry or synthesize a replacement observation.
            timing.dispatch()
            actions_issued += 1
            observation, reward, terminated, truncated, info = env.step(tensor)
            # Include completion of the returned flags in the round-trip timing.
            is_terminated = bool(_array(terminated).any())
            is_truncated = bool(_array(truncated).any())
            step_times.append(time.perf_counter() - started)
            if is_terminated or is_truncated:
                stopped = 'terminated' if is_terminated else 'truncated'
                break
            phase = 'success_feedback'
            started = time.perf_counter()
            success = env.get_wrapper_attr('is_task_success')()
            feedback_times.append(time.perf_counter() - started)
            if success:
                stopped = 'success'
                break
        timing.record(inference_times[0], model.last_inference_timing, prepare_s,
                      step_times, feedback_times, stopped)
        return observation, info, truncated, inference_times
    except Exception as exc:
        timing.event('unrecoverable_error', phase=phase,
                     attempted_actions_this_call=actions_issued,
                     completed_actions_this_call=len(step_times),
                     error_type=type(exc).__name__, error=str(exc),
                     retry_performed=False)
        raise

def reset_model(model):
    # The evaluator calls this after env.reset; do not reset the model RNG or
    # invent observations. Keep model behavior identical to the submitted policy.
    model.timing.reset()
    model._chunk_stopped = False
    # A reset does not resolve an ambiguous transport failure or cancel a robot
    # queue. Keep _chunk_failed latched until the host explicitly rebinds.
