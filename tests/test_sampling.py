import numpy as np
import pytest

from sampling import sample_token, softmax, top_candidates, top_p_filter


def test_softmax_sums_to_one():
    probs = softmax(np.array([1.0, 2.0, 3.0]))
    assert probs.sum() == pytest.approx(1.0)


def test_softmax_prefers_larger_logit():
    probs = softmax(np.array([1.0, 5.0, 2.0]))
    assert probs.argmax() == 1


def test_top_p_filter_drops_low_probability_tail():
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    filtered = top_p_filter(probs, top_p=0.8)
    assert filtered[3] == 0.0
    assert filtered[0] > 0
    assert filtered[1] > 0
    assert filtered.sum() == pytest.approx(1.0)


def test_top_p_filter_keeps_everything_when_top_p_is_one():
    probs = np.array([0.4, 0.35, 0.25])
    filtered = top_p_filter(probs, top_p=1.0)
    assert np.all(filtered > 0)


def test_top_candidates_returns_sorted_top_k():
    probs = np.array([0.1, 0.5, 0.05, 0.25, 0.08])
    candidates = top_candidates(probs, k=3)
    assert candidates[0] == (1, pytest.approx(0.5))
    assert candidates[1] == (3, pytest.approx(0.25))
    assert candidates[2] == (0, pytest.approx(0.1))


def test_sample_token_only_picks_within_top_p_set():
    logits = np.array([5.0, 4.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(42)
    for _ in range(50):
        token_id, _candidates = sample_token(logits, temperature=1.0, top_p=0.5, rng=rng)
        assert token_id in (0, 1)


def test_sample_token_low_temperature_is_greedy():
    logits = np.array([1.0, 9.0, 2.0])
    rng = np.random.default_rng(7)
    token_id, _candidates = sample_token(logits, temperature=0.01, top_p=1.0, rng=rng)
    assert token_id == 1
