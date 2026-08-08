from pathlib import Path

_ROOT = Path(__file__).resolve().parent

MODEL_PATH = _ROOT / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf"
MAX_CONTEXT_TOKENS = 4096
MAX_NEW_TOKENS = 256
CLOSER_TOKEN_MARGIN = 8
TEMPERATURE = 0.8
TOP_P = 0.9

GRPC_HOST = "localhost"
GRPC_PORT = 50051

CERT_PATH = _ROOT / "certs" / "server.crt"
KEY_PATH = _ROOT / "certs" / "server.key"
