import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cumulative, top_p) + 1)
    keep_idx = sorted_idx[:cutoff]

    filtered = np.zeros_like(probs)
    filtered[keep_idx] = probs[keep_idx]
    return filtered / filtered.sum()


def top_candidates(probs: np.ndarray, k: int) -> list[tuple[int, float]]:
    top_idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in top_idx]


def sample_token(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> tuple[int, list[tuple[int, float]]]:
    scaled = logits / max(temperature, 1e-6)
    probs = softmax(scaled)
    filtered = top_p_filter(probs, top_p)
    candidates = top_candidates(filtered, k=5)
    token_id = int(rng.choice(len(filtered), p=filtered))
    if token_id not in {c[0] for c in candidates}:
        candidates.append((token_id, float(filtered[token_id])))
    return token_id, candidates
