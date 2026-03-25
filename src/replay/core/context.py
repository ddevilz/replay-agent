from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from replay.core.recorder import Recorder

# One ContextVar, one concern.
# async-safe equivalent of thread-local storage — each task gets its own slot.
_current_recorder: ContextVar[Optional["Recorder"]] = ContextVar(
    "replay_current_recorder",
    default=None,
)


def get_recorder() -> Optional["Recorder"]:
    """Return the active Recorder for the current async task, or None."""
    return _current_recorder.get()


def set_recorder(recorder: "Recorder") -> Token["Recorder | None"]:
    """Install recorder as the active one. Returns a token for later reset."""
    return _current_recorder.set(recorder)


def reset_recorder(token: Token["Recorder | None"]) -> None:
    """Restore the previous recorder state using the token from set_recorder."""
    _current_recorder.reset(token)
