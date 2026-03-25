from __future__ import annotations

"""Smoke tests that verify the public API surface is correctly exported."""

import replay


class TestPublicAPI:
    def test_trace_is_callable(self) -> None:
        assert callable(replay.trace)

    def test_session_is_accessible(self) -> None:
        assert callable(replay.session)

    def test_record_step_is_accessible(self) -> None:
        assert callable(replay.record_step)

    def test_configure_is_callable(self) -> None:
        assert callable(replay.configure)

    def test_step_type_is_exported(self) -> None:
        from replay import StepType
        assert StepType.LLM_CALL.value == "llm_call"

    def test_all_exports_exist(self) -> None:
        for name in replay.__all__:
            assert hasattr(replay, name), f"'{name}' is in __all__ but missing from module"

    def test_configure_returns_config(self) -> None:
        from replay import ReplayConfig
        config = replay.configure(log_level="WARNING")
        assert isinstance(config, ReplayConfig)
