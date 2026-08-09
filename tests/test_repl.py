import grpc

from repl import run_turn


class FakeEvent:
    """Stand-in for a inference_pb2.GenerateEvent."""

    def __init__(self, token_id, is_eos=False, candidates=None):
        self.token_id = token_id
        self.text = ""
        self.candidates = candidates or []
        self.is_eos = is_eos


class FakeCall:
    """Iterable stand-in for the streaming call returned by stub.Generate().

    Yields `events` in order. If `exc` is given, it is raised immediately
    after the last event has been yielded (or immediately, if `events` is
    empty) -- this lets a single fake represent both a mid-stream RpcError
    and a KeyboardInterrupt raised while the caller is mid-iteration.
    """

    def __init__(self, events, exc=None):
        self._events = events
        self._exc = exc
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def __iter__(self):
        for event in self._events:
            yield event
        if self._exc is not None:
            raise self._exc


class FakeStub:
    def __init__(self, call):
        self._call = call
        self.requests = []

    def Generate(self, request):
        self.requests.append(request)
        return self._call


class FakeRpcError(grpc.RpcError):
    """A grpc.RpcError subclass that implements the .code()/.details()
    surface run_turn's except clause relies on -- the real grpc error
    classes are not constructible directly in tests."""

    def __init__(self, code, details):
        super().__init__()
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class FakeTokenizer:
    """Fake tokenizer for run_turn. Token ids are arbitrary markers --
    only their identity/count matters for the assertions, not any real
    vocabulary.

    Deliberately does NOT implement token_eos(): after Fix 2, run_turn
    must derive EOS from the stream's event.is_eos, not by re-deriving it
    from the tokenizer. If run_turn regresses to calling
    tokenizer.token_eos() again, these tests should blow up with an
    AttributeError rather than silently passing.
    """

    CLOSER_TOKENS = [9001, 9002]
    PROMPT_TOKEN = 1

    def tokenize(self, data, add_bos, special):
        if data == b"<|im_end|>\n":
            return list(self.CLOSER_TOKENS)
        return [self.PROMPT_TOKEN]

    def detokenize(self, tokens, special):
        return " ".join(str(t) for t in tokens).encode("utf-8")


def test_run_turn_excludes_eos_token_and_skips_closer():
    call = FakeCall(
        events=[
            FakeEvent(100),
            FakeEvent(101),
            FakeEvent(999, is_eos=True),
        ]
    )
    stub = FakeStub(call)
    tokenizer = FakeTokenizer()

    context_tokens, response_text, interrupted, num_generated = run_turn(
        stub, tokenizer, [10, 20], "hello"
    )

    assert interrupted is False
    assert num_generated == 2
    assert response_text == "100 101"
    # EOS token itself is appended to context (it's part of the transcript)...
    assert context_tokens[-1] == 999
    # ...but the closer must NOT be appended, since saw_eos was true.
    assert FakeTokenizer.CLOSER_TOKENS[0] not in context_tokens
    assert FakeTokenizer.CLOSER_TOKENS[1] not in context_tokens


def test_run_turn_appends_closer_when_budget_exhausted_without_eos():
    call = FakeCall(
        events=[
            FakeEvent(100),
            FakeEvent(101),
            FakeEvent(102),
        ]
    )
    stub = FakeStub(call)
    tokenizer = FakeTokenizer()

    context_tokens, response_text, interrupted, num_generated = run_turn(
        stub, tokenizer, [10, 20], "hello"
    )

    assert interrupted is False
    assert num_generated == 3
    assert response_text == "100 101 102"
    # No EOS was seen, so the closer must be appended at the end.
    assert context_tokens[-2:] == FakeTokenizer.CLOSER_TOKENS


def test_run_turn_restores_original_context_on_rpc_error():
    original_context_tokens = [10, 20]
    call = FakeCall(
        events=[FakeEvent(100)],
        exc=FakeRpcError(grpc.StatusCode.UNAVAILABLE, "server unreachable"),
    )
    stub = FakeStub(call)
    tokenizer = FakeTokenizer()

    context_tokens, response_text, interrupted, num_generated = run_turn(
        stub, tokenizer, original_context_tokens, "hello"
    )

    # Regression guard for a0327d3: on RpcError, run_turn must hand back the
    # untouched context passed in by the caller -- not a context mutated
    # with the prompt tokens / partial generation from this failed turn.
    assert context_tokens == [10, 20]
    assert response_text == ""
    assert interrupted is False
    assert num_generated == 0


def test_run_turn_handles_keyboard_interrupt_mid_stream():
    call = FakeCall(
        events=[FakeEvent(100), FakeEvent(101)],
        exc=KeyboardInterrupt(),
    )
    stub = FakeStub(call)
    tokenizer = FakeTokenizer()

    context_tokens, response_text, interrupted, num_generated = run_turn(
        stub, tokenizer, [10, 20], "hello"
    )

    assert interrupted is True
    assert call.cancelled is True
    assert num_generated == 2
    assert response_text == "100 101"


from fastapi.testclient import TestClient

from repl import app


def test_chat_websocket_returns_full_response_per_turn():
    call = FakeCall(events=[FakeEvent(100), FakeEvent(101), FakeEvent(999, is_eos=True)])
    app.state.stub = FakeStub(call)
    app.state.tokenizer = FakeTokenizer()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_text("hello")
            response = websocket.receive_json()

    assert response["type"] == "response"
    assert response["text"] == "100 101"
    assert response["interrupted"] is False
    assert response["tokens_generated"] == 2


def test_chat_websocket_ignores_blank_messages():
    call = FakeCall(events=[FakeEvent(100, is_eos=True)])
    app.state.stub = FakeStub(call)
    app.state.tokenizer = FakeTokenizer()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_text("   ")
            websocket.send_text("hello")
            response = websocket.receive_json()

    assert response["type"] == "response"
    assert response["text"] == ""


def test_chat_websocket_reuses_context_across_turns():
    call = FakeCall(events=[FakeEvent(100, is_eos=True)])
    stub = FakeStub(call)
    app.state.stub = stub
    app.state.tokenizer = FakeTokenizer()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_text("first")
            websocket.receive_json()
            websocket.send_text("second")
            websocket.receive_json()

    assert len(stub.requests) == 2
    assert len(stub.requests[1].context_tokens) > len(stub.requests[0].context_tokens)
