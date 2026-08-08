from context import truncate_context


def test_truncate_context_no_op_when_under_limit():
    tokens = [1, 2, 3]
    assert truncate_context(tokens, max_tokens=10) == [1, 2, 3]


def test_truncate_context_keeps_most_recent_tokens():
    tokens = list(range(10))
    assert truncate_context(tokens, max_tokens=4) == [6, 7, 8, 9]


def test_truncate_context_exact_limit_is_no_op():
    tokens = [1, 2, 3, 4]
    assert truncate_context(tokens, max_tokens=4) == [1, 2, 3, 4]
