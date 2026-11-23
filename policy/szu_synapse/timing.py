"""Policy-side wall-clock telemetry; no environment patching or robot commands."""
import json
import time


class EpisodeTiming:
    def __init__(self, task, load_seconds):
        self.task = task
        self.load_seconds = load_seconds
        self.episode = -1
        self.reset()

    def reset(self):
        self.episode += 1
        self.start = time.perf_counter()
        self.calls = 0
        self.actions = 0
        self.first_dispatch_s = None
        self.inference_s = 0.0
        self.step_s = 0.0
        self.feedback_s = 0.0

    def event(self, event, **fields):
        # One process emits these lines serially. Logs can be aligned with the
        # evaluator's observation/transport and hardware execution timestamps.
        print('SZU_TIMING ' + json.dumps({
            'event': event, 'task': self.task, 'episode': self.episode,
            'unix_time_s': time.time(),
            'policy_episode_elapsed_s': time.perf_counter() - self.start,
            'completed_calls': self.calls, 'completed_actions': self.actions,
            **fields,
        }, allow_nan=False), flush=True)

    def dispatch(self):
        if self.first_dispatch_s is None:
            self.first_dispatch_s = time.perf_counter() - self.start

    def record(self, inference_s, inference_detail, prepare_s, steps, feedback, stopped):
        self.calls += 1
        self.actions += len(steps)
        self.inference_s += inference_s
        self.step_s += sum(steps)
        self.feedback_s += sum(feedback)
        self.event(
            'chunk_complete', requested_mode='public_sequential_step',
            model_load_s=self.load_seconds,
            time_to_first_action_dispatch_s=self.first_dispatch_s,
            inference_total_s=inference_s, **inference_detail,
            action_preparation_s=prepare_s, step_roundtrip_s=steps,
            success_feedback_s=feedback, stop_reason=stopped,
            cumulative_inference_s=self.inference_s,
            cumulative_step_roundtrip_s=self.step_s,
            cumulative_success_feedback_s=self.feedback_s,
            # env.step is opaque: transport, execution and observation cannot
            # be separated without timestamps from the official bridge.
            observation_transport_s=None, hardware_execution_s=None,
            official_total_episode_s=None,
            clock_scope='since reset_model; excludes model load and environment reset',
        )
