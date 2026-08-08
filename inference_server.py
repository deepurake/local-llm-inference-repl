from concurrent import futures

import grpc
import numpy as np

import config
import inference_pb2
import inference_pb2_grpc
from model import load_model
from sampling import sample_token


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, model):
        self.model = model
        self.rng = np.random.default_rng()

    def Generate(self, request, context):
        context_tokens = list(request.context_tokens)

        if len(context_tokens) + request.max_new_tokens > config.MAX_CONTEXT_TOKENS:
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"context_tokens ({len(context_tokens)}) + max_new_tokens "
                f"({request.max_new_tokens}) exceeds MAX_CONTEXT_TOKENS "
                f"({config.MAX_CONTEXT_TOKENS})",
            )

        self.model.reset()
        self.model.eval(context_tokens)
        eos_id = self.model.token_eos()

        for _ in range(request.max_new_tokens):
            if not context.is_active():
                return

            logits = np.array(
                self.model.scores[self.model.n_tokens - 1, :], dtype=np.float64
            )
            token_id, candidates = sample_token(
                logits, request.temperature, request.top_p, self.rng
            )
            text = self.model.detokenize([token_id], special=True).decode(
                "utf-8", errors="replace"
            )

            yield inference_pb2.GenerateEvent(
                token_id=token_id,
                text=text,
                candidates=[
                    inference_pb2.Candidate(
                        token_id=cid,
                        text=self.model.detokenize([cid], special=True).decode(
                            "utf-8", errors="replace"
                        ),
                        probability=prob,
                    )
                    for cid, prob in candidates
                ],
                is_eos=(token_id == eos_id),
            )

            self.model.eval([token_id])

            if token_id == eos_id:
                break


def create_server(port=None, model_path=None):
    port = port or config.GRPC_PORT
    model_path = model_path or config.MODEL_PATH

    model = load_model(model_path, n_ctx=config.MAX_CONTEXT_TOKENS)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(model), server)

    with open(config.KEY_PATH, "rb") as f:
        private_key = f.read()
    with open(config.CERT_PATH, "rb") as f:
        certificate_chain = f.read()
    server_credentials = grpc.ssl_server_credentials([(private_key, certificate_chain)])

    server.add_secure_port(f"[::]:{port}", server_credentials)
    return server


def serve():
    server = create_server()
    server.start()
    print(f"Inference server listening on port {config.GRPC_PORT} (TLS)")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
