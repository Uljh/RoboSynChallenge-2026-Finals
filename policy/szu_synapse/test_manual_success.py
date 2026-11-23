"""Finals mode: omit explicit success RPCs, retain real step feedback and stops."""
from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import deploy_policy as adapter
from . import test_runtime_contract as fixtures


class NoSuccessQueries(fixtures.FeedbackFixture):
    def get_wrapper_attr(self, name):
        raise AssertionError('Finals policy must not request task success')


class ManualSuccessTests(unittest.TestCase):
    def make_model(self, task='items_handover'):
        underlying = fixtures.ModelFixture()
        with patch.object(adapter, 'load_policy', return_value=underlying):
            # Test the actual default used when the host omits new config keys.
            model = adapter.get_model({'task_name': task})
        self.assertEqual(model.success_check_mode, 'manual')
        return model, underlying

    def test_ten_tasks_keep_actions_measured_feedback_and_stop_flags(self):
        tasks = json.loads((adapter.ROOT / 'task_bindings.json').read_text())
        self.assertEqual(len(tasks), 10)
        for task in tasks:
            for stop in ('none', 'terminated', 'truncated'):
                with self.subTest(task=task, stop=stop), redirect_stdout(io.StringIO()) as log:
                    model, underlying = self.make_model(task)
                    env = NoSuccessQueries(stop)
                    result = adapter.eval(env, model, fixtures.observation(2))
                    count = 10 if stop == 'none' else 3
                    expected = underlying.actions[:count].copy()
                    expected[:, [6, 13]] = np.clip(expected[:, [6, 13]], 0, 1)
                    np.testing.assert_array_equal(torch.cat(env.issued).numpy(), expected)
                    self.assertIs(result[0], env.measured[count])
                    self.assertEqual(result[1], {'frame': count})
                    self.assertEqual(bool(result[2]), stop == 'truncated')
                    self.assertEqual(len(underlying.calls), 1)
                    self.assertEqual(env.success_calls, 0)
                    record = json.loads(log.getvalue().splitlines()[-1].removeprefix('SZU_TIMING '))
                    self.assertEqual(record['completed_actions'], count)
                    self.assertEqual(record['success_feedback_s'], [])
                    self.assertEqual(record['stop_reason'], None if stop == 'none' else stop)

    def test_manual_success_does_not_override_host_episode_control(self):
        model, _ = self.make_model()
        env = NoSuccessQueries('success')
        with redirect_stdout(io.StringIO()):
            adapter.eval(env, model, fixtures.observation(2))
        self.assertEqual(len(env.issued), 10)
        self.assertEqual(env.success_calls, 0)

    def test_next_inference_uses_returned_measured_observation(self):
        model, underlying = self.make_model()
        first = NoSuccessQueries()
        second = NoSuccessQueries()
        with redirect_stdout(io.StringIO()):
            measured, _, _, _ = adapter.eval(first, model, fixtures.observation(2))
            adapter.eval(second, model, measured)
        np.testing.assert_array_equal(underlying.calls[1]['observation/state'],
                                      measured['robot']['qpos'][0])

    def test_execution_error_is_immediate_and_not_retried(self):
        model, _ = self.make_model()
        env = NoSuccessQueries(fail_at=4)
        with redirect_stdout(io.StringIO()) as log, self.assertRaises(TimeoutError):
            adapter.eval(env, model, fixtures.observation(2))
        self.assertEqual(len(env.issued), 4)
        record = json.loads(log.getvalue().splitlines()[-1].removeprefix('SZU_TIMING '))
        self.assertEqual(record['completed_actions_this_call'], 3)
        self.assertFalse(record['retry_performed'])

    def test_yaml_uses_manual_and_simulation_opt_in_is_explicit(self):
        import yaml
        config = yaml.safe_load((adapter.ROOT / 'deploy_policy.yml').read_text())
        self.assertEqual(config['success_check_mode'], 'manual')
        model, _ = fixtures.ContractTests().make_model(success_check_mode='per_step')
        env = fixtures.FeedbackFixture('success')
        with redirect_stdout(io.StringIO()):
            adapter.eval(env, model, fixtures.observation(2))
        self.assertEqual(len(env.issued), 3)
        self.assertEqual(env.success_calls, 3)

    def test_invalid_mode_fails_before_loading_weights(self):
        with patch.object(adapter, 'load_policy') as load:
            with self.assertRaisesRegex(ValueError, 'success_check_mode'):
                adapter.get_model({'task_name': 'items_handover', 'success_check_mode': 'unknown'})
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
