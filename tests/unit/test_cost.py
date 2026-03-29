from __future__ import annotations


from replay.strategies.cost import calculate_cost


class TestCalculateCost:
    def test_known_model_returns_float(self) -> None:
        result = calculate_cost("gpt-4o", tokens_in=1000, tokens_out=500)
        assert result is not None
        assert result > 0

    def test_unknown_model_returns_none(self) -> None:
        result = calculate_cost("gpt-99-ultra", tokens_in=1000, tokens_out=500)
        assert result is None

    def test_unknown_model_never_returns_zero(self) -> None:
        # 0.0 would imply a free call; None correctly signals "unknown cost"
        result = calculate_cost("made-up-model", tokens_in=0, tokens_out=0)
        assert result is None

    def test_cached_tokens_reduce_cost(self) -> None:
        full = calculate_cost("gpt-4o", tokens_in=1000, tokens_out=0)
        cached = calculate_cost("gpt-4o", tokens_in=1000, tokens_out=0, cached_tokens_in=500)
        assert full is not None
        assert cached is not None
        assert cached < full

    def test_zero_tokens_returns_zero_cost(self) -> None:
        result = calculate_cost("gpt-4o", tokens_in=0, tokens_out=0)
        assert result == 0.0

    def test_claude_sonnet_pricing(self) -> None:
        # $0.003/1K input + $0.015/1K output
        result = calculate_cost("claude-sonnet-4-20250514", tokens_in=1000, tokens_out=1000)
        expected = 0.003 + 0.015
        assert result is not None
        assert abs(result - expected) < 1e-9
