import time

import grpc
import pytest

import config
import inference_pb2
import inference_pb2_grpc
from inference_server import create_server

TEST_PORT = 50151

# The server serializes all requests through a single worker thread
# (ThreadPoolExecutor(max_workers=1) in create_server), so a follow-up
# request's response time is an observable proxy for whether a prior
# cancelled stream actually stopped running server-side. Measured
# per-token latency on this model/hardware is ~0.03s/token, so a
# 3-token follow-up completes in well under a second when the worker
# is free. If cancellation were not honored, the follow-up would sit
# behind the remainder of the cancelled request's up-to-256-token
# budget (~7s at the measured rate) before it could even start.
# 2 seconds gives ample margin above the fast case and is well short
# of the ~7s stuck case.
FOLLOW_UP_TIMEOUT_SECONDS = 2.0


@pytest.fixture(scope="module")
def running_server():
    server = create_server(port=TEST_PORT)
    server.start()
    yield server
    server.stop(grace=1).wait()


@pytest.fixture(scope="module")
def stub(running_server):
    with open(config.CERT_PATH, "rb") as f:
        trusted_certs = f.read()
    credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)
    channel = grpc.secure_channel(f"localhost:{TEST_PORT}", credentials)
    return inference_pb2_grpc.InferenceStub(channel)


def test_generate_streams_events_for_a_short_prompt(stub):
    request = inference_pb2.GenerateRequest(
        context_tokens=[1, 2, 3],
        max_new_tokens=5,
        temperature=0.8,
        top_p=0.9,
    )
    events = list(stub.Generate(request))

    assert len(events) >= 1
    for event in events:
        assert isinstance(event.token_id, int)
        assert len(event.candidates) >= 1
        for candidate in event.candidates:
            assert 0.0 <= candidate.probability <= 1.0


def test_generate_stops_at_max_new_tokens_without_eos(stub):
    request = inference_pb2.GenerateRequest(
        context_tokens=[1, 2, 3],
        max_new_tokens=3,
        temperature=0.8,
        top_p=0.9,
    )
    events = list(stub.Generate(request))

    assert len(events) <= 3


def test_generate_rejects_context_overflow(stub):
    oversized_context = list(range(config.MAX_CONTEXT_TOKENS))
    request = inference_pb2.GenerateRequest(
        context_tokens=oversized_context,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=0.8,
        top_p=0.9,
    )

    with pytest.raises(grpc.RpcError) as exc_info:
        list(stub.Generate(request))

    assert exc_info.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_generate_rejects_empty_context(stub):
    request = inference_pb2.GenerateRequest(
        context_tokens=[],
        max_new_tokens=5,
        temperature=0.8,
        top_p=0.9,
    )

    with pytest.raises(grpc.RpcError) as exc_info:
        list(stub.Generate(request))

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_generate_stops_on_client_cancellation(stub):
    request = inference_pb2.GenerateRequest(
        context_tokens=[1, 2, 3],
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=0.8,
        top_p=0.9,
    )
    call = stub.Generate(request)
    first_event = next(iter(call))
    assert first_event is not None

    call.cancel()

    with pytest.raises(grpc.RpcError) as exc_info:
        list(call)
    assert exc_info.value.code() == grpc.StatusCode.CANCELLED

    # The client-side CANCELLED error above is true regardless of whether the
    # server actually stopped generating in the background. Since the server
    # serializes all requests through a single worker thread (max_workers=1),
    # a cancelled stream that kept running server-side would block this
    # follow-up behind its remaining token budget. A fast follow-up is
    # therefore server-observable proof that cancellation was honored.
    follow_up_request = inference_pb2.GenerateRequest(
        context_tokens=[1, 2, 3],
        max_new_tokens=3,
        temperature=0.8,
        top_p=0.9,
    )
    start = time.monotonic()
    follow_up_events = list(stub.Generate(follow_up_request))
    elapsed = time.monotonic() - start

    assert len(follow_up_events) >= 1
    assert elapsed < FOLLOW_UP_TIMEOUT_SECONDS
