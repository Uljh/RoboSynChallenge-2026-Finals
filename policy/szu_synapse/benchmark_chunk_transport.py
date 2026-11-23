"""Wall-clock delay injection, not an official network or robot benchmark."""
from contextlib import redirect_stdout
import io
import json
import time

from . import deploy_policy as adapter
from .chunk_execution import bind_chunk_executor
from .test_chunk_execution import OrderedFixture
from .test_runtime_contract import ContractTests, FeedbackFixture, observation


def run():
    results = []
    factory = ContractTests()
    for delay in (.1, .3, .5):
        for remote_success in (False, True):
            env = FeedbackFixture()
            original_step = env.step
            original_attr = env.get_wrapper_attr
            requests = [0]
            def delayed_step(action):
                requests[0] += 1
                time.sleep(delay)
                return original_step(action)
            def delayed_attr(name):
                fn = original_attr(name)
                def call():
                    if remote_success:
                        requests[0] += 1
                        time.sleep(delay)
                    return fn()
                return call
            env.step = delayed_step
            env.get_wrapper_attr = delayed_attr
            model, _ = factory.make_model()
            with redirect_stdout(io.StringIO()):
                start = time.perf_counter()
                adapter.eval(env, model, observation(2))
                sequential = time.perf_counter()-start
            executor = OrderedFixture(FeedbackFixture())
            original = executor.execute
            def delayed_chunk(request):
                time.sleep(delay)
                return original(request)
            executor.execute = delayed_chunk
            model, _ = factory.make_model()
            bind_chunk_executor(model, executor)
            with redirect_stdout(io.StringIO()):
                start = time.perf_counter()
                adapter.eval(None, model, observation(2))
                batched = time.perf_counter()-start
            assert len(env.issued) == len(executor.env.issued) == 10
            results.append(dict(delay_s=delay, success_query_remote=remote_success,
                                sequential_requests=requests[0], chunk_requests=len(executor.requests),
                                sequential_wall_s=sequential, chunk_wall_s=batched))
    print(json.dumps(dict(scope='Actual sleep-injected transport fixture; mock model; no physical execution duration',
                          results=results), indent=2))


if __name__ == '__main__':
    run()
