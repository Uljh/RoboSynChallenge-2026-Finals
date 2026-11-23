# Finals timeout correction — 23 November 2026

Evaluation: `real_7957b8d0bf6047`, team `finalist_18`. This correction covers all
ten registered tasks sharing `policy/szu_synapse`.

## Change to use for evaluation

Use the existing task commands, weights and policy entry. The model default and
`deploy_policy.yml` now set `success_check_mode: manual`. This removes the
policy's per-action `get_wrapper_attr("is_task_success")()` calls. The organizer
FAQ states that physical success/episode completion is handled manually.

Every issued action still uses the existing `env.step` interface. The policy
checks returned `terminated` and `truncated` flags after each action and stops
immediately on either. It returns the final measured observation unchanged for
the next prediction. No commanded state is substituted for feedback. Exceptions
are reported with their phase and attempted/completed counts, then raised
without retry. The host remains responsible for manual success/end-of-episode
control, including halting execution through its existing environment/SDK.

The model still predicts 50 actions and executes the first 10, with earlier
termination/truncation honored. There is no new mid-chunk inference or history
processing. Checkpoints, normalization, task bindings, dependencies, model RNG,
action conversion and environment limits are unchanged. The only new deployment
setting is `success_check_mode`; no changes to registered task commands or HF
locations are required. All runtime changes are under `policy/`.

For local simulation, append `--success_check_mode per_step` to retain the old
per-action success checks. Mode selection is explicit; the policy does not
probe the host type or guess private robot methods.

The optional executor described in `CHUNK_EXECUTION.md` is **not enabled or
required** by this correction. No official chunk RPC integration is claimed.
The default path continues to call `env.step` per action; this change removes
additional success polling, not all possible transport round trips.

## Validation

From the repository root in the existing evaluation Python environment:

```bash
python -m unittest policy.szu_synapse.test_runtime_contract policy.szu_synapse.test_chunk_execution policy.szu_synapse.test_manual_success -v
```

All 20 test methods passed on 194 CPU. Manual-mode tests cover all ten tasks
with normal execution, termination and truncation (30 task/stop combinations),
zero success-predicate access, exact action order, final measured feedback,
next-prediction input, error-after-dispatch handling and configuration defaults.
The explicit simulation mode retains its original success stopping behavior.

All ten actual frozen checkpoints were reloaded and tested on CPU with fixed
synthetic observations and mock environment feedback. Complete 50-by-14
predictions match the original frozen baseline. Normal, terminated and truncated
execution prefixes also match. In the synthetic success-only case, manual mode
intentionally continues the existing prefix instead of consulting the simulation
success predicate; host stop flags still stop execution. Every tested manual
call made zero explicit success queries. Per-task hashes, timings and source
fingerprints are in `manual_success_evidence_20261123.json`.

These are inference and adapter regression tests, not physical task scores.
The earlier native RTX4090 handover simulation is historical evidence from
commit `6e20dda245f6d66030698b31900395872171958c`, not a real-robot validation of
this new mode. Historical diagnostic/chunk fixtures remain separately recorded
in `correction_evidence_20261123.json` and `chunk_evidence_20261123.json`.

## Timing and untested cases

`SZU_TIMING` logs model loading, reset-to-first-action dispatch, synchronized
inference, observation encoding, action preparation, per-step round trips,
success-poll timings (empty in manual mode), and policy episode elapsed time.
The model sampling timer is synchronized through host output materialization.
Policy episode timing starts at `reset_model`, after environment reset; it
excludes environment creation/reset and model loading. It does not measure
remote acceptance of the first command.

The supplied `env.step` is opaque. Observation/transport, hardware execution and
full official episode time cannot be separately measured from it; unavailable
fields stay null rather than being inferred from model time. The evaluation
host should correlate policy logs with its bridge/robot timestamps.

Untested for all ten tasks: official robot transport and manual stop propagation,
RTX5090 timing, physical end-to-end runtime and task success. The organizer's
reported items_handover episode 14 (600 seconds / 690 issued actions) has not
been reproduced. This is a conservative polling reduction for organizer
acceptance testing, not a claim that the 600-second timeout is resolved.
