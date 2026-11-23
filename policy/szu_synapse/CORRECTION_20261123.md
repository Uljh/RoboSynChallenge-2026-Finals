# Timeout investigation — 23 November 2026

Evaluation: `real_7957b8d0bf6047`. The organizer reported 600 seconds and 690
issued actions for `items_handover`, episode 14. All ten submitted tasks use
this adapter and are covered by the changes below.

## Changes

- Export `get_model`, `eval` and `reset_model` from the policy package so the
  public evaluator can import the documented entry point.
- Measure OpenPI sampling after device results are materialized on the host.
- Stage the existing ten-action prefix with one device transfer.
- Emit `SZU_TIMING` JSON lines for synchronized inference, observation encoding,
  action preparation, each `env.step` round trip, and success-feedback checks.
- Report an exception immediately with the failed phase and attempted/completed
  action counts. Do not retry commands that might already have executed.
- Separate prediction from execution and expose the unchanged ten-action
  prefix. Add an explicitly bound ordered executor with one dispatch per prefix,
  measured-observation validation, stop handling and latched execution failures.
  The host integration contract is in [CHUNK_EXECUTION.md](CHUNK_EXECUTION.md).

Weights, normalization, task bindings, requirements, prediction horizon (50),
execution prefix (10), model RNG behavior and environment limits are unchanged.
The changes are entirely under `policy/`.

## Timing boundaries

`time_to_first_action_dispatch_s` starts at `reset_model` and ends immediately
before the first `env.step` or bound executor call; it does not measure remote command acceptance.
`policy_episode_elapsed_s` uses the same start and is emitted every chunk.
The public evaluator calls `reset_model` after environment reset, so these
fields exclude model loading and environment reset. Model construction has a
separate `model_load_s` field. The evaluation host must supply the full official
episode duration.

`model_inference_s` includes input transforms, sampling, host synchronization
and output transforms. `sampling_s` is the synchronized sampling subset.
`observation_encode_s` measures conversion of an already received observation;
it is not camera capture or network transport latency.

`step_roundtrip_s` includes whatever the supplied `env.step` does: execution,
feedback, observation production and transport. These components cannot be
honestly separated through the public API, so `observation_transport_s`,
`hardware_execution_s` and `official_total_episode_s` are null. The organizer's
bridge timestamps are needed to complete that breakdown.

With a bound executor, `chunk_roundtrip_s` covers the complete request. Optional
`observation_transport_s` and `execution_feedback_s` come from that executor's
instrumentation, not estimates from the round-trip duration. Unknown fields
remain null; they must not be added to the round-trip time as independent costs.

## Outstanding interface question

Without an explicitly bound executor, this candidate **still calls the public
single-action `env.step`**. The new chunk path is implemented and tested against
an ordered execution fixture, but no official robot transport is wired to it.
The organizer's notice does not specify the SDK method, feedback or cancellation
semantics. A verified bridge implementing `CHUNK_EXECUTION.md` is still needed
before the new path can reduce real-network round trips. No guessed SDK method
or automatic interface detection is used.

## Reproduction

In the existing evaluation Python environment, from the repository root:

```bash
python -m unittest policy.szu_synapse.test_runtime_contract policy.szu_synapse.test_chunk_execution -v
python -m policy.szu_synapse.benchmark_chunk_transport
```

The contract suite exercises all ten task bindings across normal execution,
success, termination and truncation; it also checks observation layout,
invalid predictions, episode reset and a feedback error after command issue.
All 14 test methods passed. The chunk suite adds per-task sequential/chunk
comparisons, stale/malformed responses, partial errors, unknown completion,
measured next observations and stop/failure latches. These tests use CPU tensors
and fixtures, not a physical robot or network.

Latency injection exercises 10-action prefixes with 0.1/0.3/0.5 seconds per
request. The fixture's client request count falls from 10 (or 20 when success
feedback is also remote) to one. It excludes physical execution time and does
not measure the official network. Raw results are in
`chunk_evidence_20261123.json`.

All ten actual frozen checkpoints passed CPU regression against the original
fixed-fixture baseline: the complete 50-by-14 predictions and emitted action
prefixes matched exactly. This is an offline contract check with synthetic
observations and test feedback, not physical task evaluation.

On the preceding diagnostic commit `6e20dda245f6d66030698b31900395872171958c`,
one native `items_handover` run on 194 / RTX 4090 completed 350 steps / 35 calls
and stopped at the unchanged simulator step limit (task success: false).
Cache was disabled. Environment creation took 39.41 s, model loading 13.11 s,
and reset-to-observation 0.62 s. First action dispatch after reset took 42.41 s;
subsequent synchronized inference had median 0.108 s; cumulative step round
trips took 18.10 s. The reset-return-to-end episode time was 64.35 s. Policy
telemetry agreed with the independent outer timer within 0.002 s. These are
local simulation measurements, not evidence of reduced real-network latency.
GPU0 was returned to its reservation manager after the run.

See `correction_evidence_20261123.json` for per-task bindings and measurements.
Physical-network batching, RTX 5090 timing, the organizer's episode 14 and
real-robot end-to-end completion for all ten tasks remain untested. This
candidate is not ready to claim a real-transport timeout fix until the host
executor is integrated and verified. No checkpoint, dependency or task-binding
change is required by the new adapter boundary.
