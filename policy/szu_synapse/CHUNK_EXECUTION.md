# Ordered action-chunk integration

**Experimental, not enabled by the finals correction.** The submitted correction
uses the existing `env.step` entry with `success_check_mode=manual`; no executor
binding is required. This optional interface has only fixture-based transport
tests and is not an implementation of the official robot bridge.

The policy can now provide its ten-action execution prefix in one call. The
robot host must connect this to its supported ordered-execution interface.
This document defines our adapter boundary; it does not name or implement an
unpublished RoboSyn robot API. The public simulator continues to use `env.step`.

## Host binding

After constructing the policy, call
`bind_chunk_executor(model, executor)` from `policy.szu_synapse`. The host-supplied
executor declares `contract = ChunkContract()` and implements
`execute(request: ActionChunk) -> ChunkResult`; the types are in
`policy.szu_synapse.chunk_execution`. Binding is explicit. The policy does not
probe for guessed method names on the environment.

Alternatively, `predict_action_chunk(model, measured_observation)` returns the
same read-only NumPy `float32` prefix for a host that owns its execution loop.
That function issues no commands. The bound path includes response validation,
timing and error handling and is the preferred integration point.

`ActionChunk` contains a unique `request_id`, task name, an ordered `[T, 14]`
array and `control_hz=10`. By default T is 10, from the existing 50-step model
prediction. The action layout and units are unchanged: left arm six joints,
left gripper, right arm six joints, right gripper; joints are absolute radians
and grippers are in `[0, 1]`.

## Executor obligations

1. Send the entire prefix through the supported transport in one request.
   Execute actions in order at the original control rate. A client-side loop
   around ten remote `step` calls does not implement this optimization.
2. Retain task feedback dependencies on the execution side. Check the actual
   task's success, termination and truncation conditions between actions and
   cancel the unexecuted suffix on any stop. `info['success']` is not a portable
   substitute for `is_task_success()` in the public tasks.
3. Return `ChunkResult` with the matching request ID, completed-action count,
   final **measured** observation, info and scalar stop flags. A short prefix
   requires an explicit stop flag. Never construct qpos or camera observations
   from commanded targets. The next prediction uses the returned observation.
4. Keep the host environment's stop state consistent with those returned flags,
   so its outer evaluation loop ends the episode correctly. The policy refuses
   another dispatch after a reported stop until the host resets the environment
   and calls `reset_model`. The public `eval` return tuple is unchanged.
5. Use bounded transport/feedback waits and the SDK's cancellation mechanism.
   On failure, stop the remaining queue and raise `ChunkExecutionError` with
   the request ID and confirmed completed count, or `None` if unknown. Do not
   report cancellation as successful unless it has been confirmed. A Python
   timeout alone does not cancel commands on a robot.
6. Reconcile ambiguous execution and cancel pending commands before rebinding
   after an error. The adapter never retries or switches to sequential execution
   after a batch dispatch. Failure stays latched across `reset_model`.

An executor declaration alone does not verify these obligations. In particular,
the adapter cannot remotely cancel an unknown SDK queue or infer measured state.
Those behaviors must be checked on the host bridge before enabling this path.

## Timing and tests

Each bound call emits synchronized model inference, first dispatch and whole
chunk round-trip timing. The executor can supply `observation_transport_s` and
`execution_feedback_s` from its own instrumentation; otherwise they remain
null. Full episode duration, including reset and host overhead, must be measured
by the evaluator. Timing scopes are detailed in `CORRECTION_20261123.md`.

From the repository root in the evaluation Python environment:

```bash
python -m unittest policy.szu_synapse.test_runtime_contract policy.szu_synapse.test_chunk_execution -v
python -m policy.szu_synapse.benchmark_chunk_transport
```

The tests cover all ten bindings and normal/success/termination/truncation
paths, action order, measured observations, partial execution errors, unknown
completion, malformed/stale responses and stop/failure latches. The benchmark
injects 0.1/0.3/0.5 seconds of latency per request into a test transport. Its
server-side fixture performs local steps; it is not a real robot bridge and
must not be used as the submitted executor. Results measure the request-count
effect, not physical episode performance.
