import grpc
import pytest

import config
import inference_pb2
import inference_pb2_grpc
from inference_server import create_server

TEST_PORT = 50151


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
