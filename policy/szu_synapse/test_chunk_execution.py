"""Transport contract tests. Fixtures are not an official robot implementation."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from . import deploy_policy as adapter
from .chunk_execution import ChunkContract, ChunkResult, ChunkExecutionError, bind_chunk_executor
from . import test_runtime_contract as fixtures
from .test_runtime_contract import FeedbackFixture, observation


class OrderedFixture:
    contract = ChunkContract()

    def __init__(self, env):
        self.env = env
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        # This loop represents robot-side execution, not client-side RPC calls.
        try:
            for count, action in enumerate(request.actions, 1):
                obs, _, term, trunc, info = self.env.step(torch.tensor(action)[None])
                success = False if bool(term) or bool(trunc) else self.env.get_wrapper_attr('is_task_success')()
                if term or trunc or success:
                    break
        except Exception as exc:
            raise ChunkExecutionError(str(exc), request_id=request.request_id,
                                      executed_steps=len(self.env.issued)-1) from exc
        return ChunkResult(request.request_id, count, obs, info, bool(term), bool(trunc), bool(success))


class ChunkTests(unittest.TestCase):
    make_model = fixtures.ContractTests.make_model

    def test_all_tasks_sequential_and_chunk_match(self):
        tasks = json.loads((Path(adapter.__file__).parent/'task_bindings.json').read_text())
        for task in tasks:
            for stop in ('none', 'success', 'terminated', 'truncated'):
                with self.subTest(task=task, stop=stop), redirect_stdout(io.StringIO()):
                    a, _ = self.make_model(task)
                    b, underlying = self.make_model(task)
                    left, right = FeedbackFixture(stop), FeedbackFixture(stop)
                    executor = OrderedFixture(right)
                    bind_chunk_executor(b, executor)
                    initial = observation(2)
                    expected = adapter.eval(left, a, initial)
                    result = adapter.eval(None, b, initial)  # No client env.step on this path.
                    np.testing.assert_array_equal(torch.cat(left.issued), torch.cat(right.issued))
                    self.assertIs(result[0], right.measured[len(right.issued)])
                    self.assertEqual(result[1], expected[1])
                    self.assertEqual(bool(result[2]), bool(expected[2]))
                    self.assertEqual(len(executor.requests), 1)
                    self.assertEqual(len(underlying.calls), 1)
                    self.assertEqual(executor.requests[0].actions.shape, (10, 14))
                    self.assertEqual(b.timing.actions, len(right.issued))

    def test_partial_error_never_retries_or_falls_back(self):
        model, _ = self.make_model()
        env = FeedbackFixture(fail_at=4)
        executor = OrderedFixture(env)
        bind_chunk_executor(model, executor)
        with redirect_stdout(io.StringIO()) as log, self.assertRaises(ChunkExecutionError) as caught:
            adapter.eval(None, model, observation(2))
        self.assertEqual(caught.exception.executed_steps, 3)
        self.assertEqual(len(env.issued), 4)
        self.assertEqual(len(executor.requests), 1)
        self.assertIn('"sequential_fallback_performed": false', log.getvalue())

    def test_invalid_or_stale_response_is_fatal(self):
        changes = [dict(request_id='stale'), dict(executed_steps=4), dict(executed_steps=11),
                   dict(executed_steps=True), dict(observation=None), dict(success=[False]),
                   dict(execution_feedback_s=float('nan'))]
        for change in changes:
            with self.subTest(change=change):
                model, _ = self.make_model()
                executor = OrderedFixture(FeedbackFixture())
                original = executor.execute
                executor.execute = lambda req: replace(original(req), **change)
                bind_chunk_executor(model, executor)
                with redirect_stdout(io.StringIO()), self.assertRaises(ChunkExecutionError):
                    adapter.eval(None, model, observation(2))
                self.assertEqual(len(executor.requests), 1)

    def test_timeout_unknown_completion_stays_unknown(self):
        model, _ = self.make_model()
        executor = OrderedFixture(FeedbackFixture())
        def lost_reply(request):
            executor.requests.append(request)
            raise TimeoutError('reply lost after dispatch')
        executor.execute = lost_reply
        bind_chunk_executor(model, executor)
        with redirect_stdout(io.StringIO()), self.assertRaises(ChunkExecutionError) as caught:
            adapter.eval(None, model, observation(2))
        self.assertIsNone(caught.exception.executed_steps)
        self.assertEqual(len(executor.requests), 1)

    def test_next_prediction_uses_measured_state_and_unique_id(self):
        model, underlying = self.make_model()
        executor = OrderedFixture(FeedbackFixture())
        bind_chunk_executor(model, executor)
        with redirect_stdout(io.StringIO()):
            final, _, _, _ = adapter.eval(None, model, observation(2))
            executor.env = FeedbackFixture()
            adapter.reset_model(model)
            adapter.eval(None, model, final)
        np.testing.assert_array_equal(underlying.calls[1]['observation/state'], final['robot']['qpos'][0])
        self.assertNotEqual(executor.requests[0].request_id, executor.requests[1].request_id)

    def test_contract_must_be_explicit(self):
        model, _ = self.make_model()
        executor = OrderedFixture(FeedbackFixture())
        executor.contract = replace(ChunkContract(), control_hz=50)
        with self.assertRaises(ValueError):
            bind_chunk_executor(model, executor)

    def test_stop_blocks_next_dispatch_until_environment_reset(self):
        for stop in ('success', 'terminated', 'truncated'):
            with self.subTest(stop=stop), redirect_stdout(io.StringIO()):
                model, underlying = self.make_model()
                executor = OrderedFixture(FeedbackFixture(stop))
                bind_chunk_executor(model, executor)
                adapter.eval(None, model, observation(2))
                with self.assertRaisesRegex(RuntimeError, 'episode stop'):
                    adapter.eval(None, model, observation(2))
                self.assertEqual(len(executor.requests), 1)
                self.assertEqual(len(underlying.calls), 1)
                executor.env = FeedbackFixture()
                adapter.reset_model(model)
                adapter.eval(None, model, observation(2))
                self.assertEqual(len(executor.requests), 2)

    def test_failure_stays_latched_across_reset(self):
        model, _ = self.make_model()
        executor = OrderedFixture(FeedbackFixture(fail_at=4))
        bind_chunk_executor(model, executor)
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ChunkExecutionError):
                adapter.eval(None, model, observation(2))
            adapter.reset_model(model)
            with self.assertRaisesRegex(RuntimeError, 'Previous chunk failed'):
                adapter.eval(None, model, observation(2))
        self.assertEqual(len(executor.requests), 1)

    def test_invalid_final_observation_preserves_execution_count(self):
        model, _ = self.make_model()
        executor = OrderedFixture(FeedbackFixture())
        original = executor.execute
        executor.execute = lambda req: replace(original(req), observation={})
        bind_chunk_executor(model, executor)
        with redirect_stdout(io.StringIO()) as log, self.assertRaises(KeyError):
            adapter.eval(None, model, observation(2))
        record = json.loads(log.getvalue().splitlines()[-1].removeprefix('SZU_TIMING '))
        self.assertEqual(record['completed_actions_this_call'], 10)
        self.assertEqual(record['request_id'], executor.requests[0].request_id)
        self.assertTrue(model._chunk_failed)


if __name__ == '__main__':
    unittest.main()
