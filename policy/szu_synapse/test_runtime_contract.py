"""CPU contract tests. No checkpoints, robot connection or simulated success claims.

Run: python -m unittest policy.szu_synapse.test_runtime_contract -v
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import deploy_policy as adapter


def observation(seed):
    rng = np.random.default_rng(seed)
    return {
        'robot': {'qpos': rng.uniform(-.1, .1, (1, 14)).astype(np.float32)},
        'sensor': {camera: {'color': rng.integers(0, 256, (1, 8, 8, 3), dtype=np.uint8)}
                   for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')},
    }


class ModelFixture:
    def __init__(self):
        self.actions = np.random.default_rng(5).normal(size=(50, 14)).astype(np.float32)
        self.calls = []
        self.rng_marker = object()

    def infer(self, obs, noise=None):
        self.calls.append(obs)
        return {'actions': self.actions.copy(), 'policy_timing': {'infer_ms': 1.0}}


class FeedbackFixture:
    def __init__(self, stop='none', stop_at=3, fail_at=None):
        self.unwrapped = SimpleNamespace(device='cpu', single_action_space=SimpleNamespace(shape=(14,)))
        self.stop, self.stop_at, self.fail_at = stop, stop_at, fail_at
        self.issued = []
        self.measured = [observation(100+i) for i in range(11)]
        self.success_calls = 0

    def step(self, action):
        self.issued.append(action.clone())
        index = len(self.issued)
        if index == self.fail_at:
            raise TimeoutError('fixture: feedback unavailable after command submission')
        self_index = index == self.stop_at
        return (self.measured[index], 0, torch.tensor([self_index and self.stop == 'terminated']),
                torch.tensor([self_index and self.stop == 'truncated']), {'frame': index})

    def get_wrapper_attr(self, name):
        if name != 'is_task_success':
            raise AttributeError(name)
        def success():
            self.success_calls += 1
            return self.stop == 'success' and len(self.issued) == self.stop_at
        return success


class ContractTests(unittest.TestCase):
    def make_model(self, task='items_handover', **args):
        underlying = ModelFixture()
        args.setdefault('success_check_mode', 'per_step')
        with patch.object(adapter, 'load_policy', return_value=underlying):
            model = adapter.get_model(dict(task_name=task, **args))
        return model, underlying

    def test_all_ten_bindings_prefix_order_feedback_and_stops(self):
        tasks = json.loads((Path(adapter.__file__).parent / 'task_bindings.json').read_text())
        self.assertEqual(len(tasks), 10)
        for task in tasks:
            for stop in ('none', 'terminated', 'truncated', 'success'):
                with self.subTest(task=task, stop=stop), redirect_stdout(io.StringIO()) as log:
                    model, underlying = self.make_model(task)
                    env = FeedbackFixture(stop)
                    initial = observation(2)
                    returned, info, truncated, times = adapter.eval(env, model, initial)
                    expected_steps = 10 if stop == 'none' else 3
                    self.assertEqual(len(env.issued), expected_steps)
                    expected = underlying.actions[:expected_steps].copy()
                    expected[:, [6, 13]] = np.clip(expected[:, [6, 13]], 0, 1)
                    np.testing.assert_array_equal(torch.cat(env.issued).numpy(), expected)
                    self.assertIs(returned, env.measured[expected_steps])
                    self.assertEqual(info, {'frame': expected_steps})
                    self.assertEqual(bool(truncated), stop == 'truncated')
                    self.assertEqual(len(underlying.calls), 1)
                    self.assertEqual(underlying.calls[0]['prompt'], tasks[task]['prompt'])
                    np.testing.assert_array_equal(underlying.calls[0]['state'] if 'state' in underlying.calls[0]
                                                  else underlying.calls[0]['observation/state'], initial['robot']['qpos'][0])
                    self.assertEqual(len(times), 1)
                    record = json.loads(log.getvalue().splitlines()[-1].removeprefix('SZU_TIMING '))
                    self.assertEqual(record['completed_actions'], expected_steps)
                    self.assertEqual(len(record['step_roundtrip_s']), expected_steps)
                    self.assertIsNotNone(record['time_to_first_action_dispatch_s'])
                    self.assertIsNone(record['official_total_episode_s'])
                    self.assertIsNone(record['hardware_execution_s'])

    def test_command_accepted_then_feedback_error_is_not_retried(self):
        model, _ = self.make_model()
        env = FeedbackFixture(fail_at=4)
        with redirect_stdout(io.StringIO()) as log:
            with self.assertRaisesRegex(TimeoutError, 'feedback unavailable'):
                adapter.eval(env, model, observation(2))
        self.assertEqual(len(env.issued), 4)
        record = json.loads(log.getvalue().splitlines()[-1].removeprefix('SZU_TIMING '))
        self.assertEqual(record['phase'], 'env.step')
        self.assertEqual(record['attempted_actions_this_call'], 4)
        self.assertEqual(record['completed_actions_this_call'], 3)
        self.assertFalse(record['retry_performed'])

    def test_reset_preserves_policy_and_resets_only_timing(self):
        model, underlying = self.make_model()
        with redirect_stdout(io.StringIO()):
            adapter.eval(FeedbackFixture(), model, observation(2))
        rng_marker = underlying.rng_marker
        adapter.reset_model(model)
        self.assertIs(model.policy, underlying)
        self.assertIs(underlying.rng_marker, rng_marker)
        self.assertEqual(model.timing.actions, 0)
        self.assertIsNone(model.timing.first_dispatch_s)

    def test_invalid_prediction_never_issues_a_command(self):
        for kind in ('shape', 'nan'):
            with self.subTest(kind=kind):
                model, underlying = self.make_model()
                if kind == 'shape':
                    underlying.actions = np.zeros((49, 14), dtype=np.float32)
                else:
                    underlying.actions[0, 0] = np.nan
                env = FeedbackFixture()
                with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    adapter.eval(env, model, observation(2))
                self.assertEqual(env.issued, [])

    def test_32d_state_and_camera_order(self):
        obs = observation(2)
        obs['robot']['qpos'] = np.arange(32, dtype=np.float32)[None]
        obs['sensor']['cam_high']['color'].fill(11)
        obs['sensor']['cam_left_wrist']['color'].fill(22)
        obs['sensor']['cam_right_wrist']['color'].fill(33)
        encoded = adapter.encode_observation(obs, 'test')
        np.testing.assert_array_equal(encoded['observation/state'], np.r_[0:7, 16:23])
        for key, value in [('image', 11), ('left_wrist_image', 22), ('right_wrist_image', 33)]:
            self.assertTrue(np.all(encoded['observation/'+key] == value))


if __name__ == '__main__':
    unittest.main()
