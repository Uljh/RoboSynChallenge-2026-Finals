"""Explicit integration contract, NOT an assumed RoboSyn robot API.

The host must bind a reviewed executor. Merely defining a method on env does
not opt into batching. A submitted chunk must never fall back to step().
"""
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any
import uuid

import numpy as np


@dataclass(frozen=True)
class ChunkContract:
    version: int = 1
    ordered_execution: bool = True
    control_hz: int = 10
    measured_final_observation: bool = True
    stop_on_success_termination_truncation: bool = True
    cancel_remaining_on_error: bool = True


@dataclass(frozen=True)
class ActionChunk:
    request_id: str
    task: str
    actions: np.ndarray  # [T,14], absolute joints in radians; grippers in [0,1]
    control_hz: int = 10


@dataclass(frozen=True)
class ChunkResult:
    request_id: str
    executed_steps: int
    observation: Any  # Final measured observation, never commanded state.
    info: Any
    terminated: bool = False
    truncated: bool = False
    success: bool = False  # Separate from info['success']; task semantics differ.
    observation_transport_s: float | None = None
    execution_feedback_s: float | None = None


class ChunkExecutionError(RuntimeError):
    """Execution may be partial. Do not retry, resume, or use sequential fallback."""
    def __init__(self, message, *, request_id=None, executed_steps=None):
        super().__init__(message)
        self.request_id = request_id
        self.executed_steps = executed_steps


def bind_chunk_executor(model, executor):
    """Host opt-in after verifying its actual transport and stop semantics.

executor.contract must be ChunkContract() and executor.execute(request) must
return ChunkResult. The declaration is not proof of hardware compliance.
After a failure, reconcile executed actions and cancel any pending queue before
rebinding. reset_model alone cannot resolve unknown execution state.
"""
    if getattr(executor, 'contract', None) != ChunkContract():
        raise ValueError('Executor does not declare the required chunk contract')
    if not callable(getattr(executor, 'execute', None)):
        raise ValueError('Executor must expose execute(ActionChunk)')
    model.chunk_executor = executor
    model._chunk_failed = False


def execute_chunk(executor, task, actions):
    values = np.array(actions, dtype=np.float32, copy=True, order='C')
    if (values.ndim != 2 or values.shape[1] != 14 or not 1 <= len(values) <= 50
            or not np.isfinite(values).all()):
        raise ValueError('Expected a finite [1..50,14] action prefix')
    if np.any(values[:, [6, 13]] < 0) or np.any(values[:, [6, 13]] > 1):
        raise ValueError('Gripper commands must be in [0,1]')
    values.setflags(write=False)
    request = ActionChunk(uuid.uuid4().hex, task, values)
    try:
        result = executor.execute(request)  # Exactly one dispatch; never retry.
        if not isinstance(result, ChunkResult) or result.request_id != request.request_id:
            raise ValueError('Wrong response type or mismatched request ID')
        count = result.executed_steps
        if isinstance(count, bool) or not isinstance(count, Integral) or not 0 <= count <= len(values):
            raise ValueError('Invalid executed-step count')
        for flag in (result.terminated, result.truncated, result.success):
            if not isinstance(flag, (bool, np.bool_)):
                raise ValueError('Stop flags must be scalar booleans')
        if count < len(values) and not (result.terminated or result.truncated or result.success):
            raise ValueError('Partial execution without an explicit stop condition')
        if result.observation is None:
            raise ValueError('Final measured observation is missing')
        for value in (result.observation_transport_s, result.execution_feedback_s):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
                raise ValueError('Invalid executor timing')
        return result
    except Exception as exc:
        # Ordinary exceptions provide no trustworthy completed-action count.
        count = None
        if isinstance(exc, ChunkExecutionError) and exc.request_id == request.request_id:
            value = exc.executed_steps
            if isinstance(value, Integral) and not isinstance(value, bool) and 0 <= value <= len(values):
                count = int(value)
        raise ChunkExecutionError(str(exc), request_id=request.request_id,
                                  executed_steps=count) from exc
