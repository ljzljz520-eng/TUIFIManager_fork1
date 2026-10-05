"""
Hosting primitives for TUIFIManager.

ManagerContext  - instance-owned state: filesystem base directory, copy
                  buffer, markers, ordering, mouse-event cache and the
                  prompt sink.
TerminalHost    - boundary for terminal side effects (raw writes, leaving
                  the curses session, the warnings hook and the process
                  working directory).
SignalBroker    - boundary for process signal installation.

Only the standalone CLI shell uses ProcessTerminalHost / ProcessSignalBroker,
which keep the historical behavior of patching warnings.showwarning, changing
the working directory and installing SIGINT/SIGTSTP handlers.

Embedded hosts get EmbeddedTerminalHost / NullSignalBroker: they never patch
warnings, never call os.chdir and never install process signal handlers.

Process-level resources are installed through opaque tokens and are restored
only by the exact instance that registered them (see TUIFIManager.dispose).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import os
import signal
import sys
import warnings

import unicurses

from .TUItilities import HOME_DIR, MouseEventCache


# ======================== ======================== ========================
#                            TerminalHost
# ======================== ======================== ========================
class TerminalHost:
    """Boundary for terminal-level side effects.

    writer        : callable receiving raw text (defaults to stdout).
    suspender     : optional callable replacing unicurses.endwin() when a
                    subprocess needs the terminal.
    restorer      : optional callable replacing unicurses.doupdate() after
                    the subprocess finishes.
    owns_warnings : allow install_warning_handler() to patch
                    warnings.showwarning.
    owns_cwd      : allow set_working_dir() to call os.chdir().
    """

    def __init__(self,
                 writer: Optional[Callable[[str], None]] = None,
                 suspender: Optional[Callable[[], Any]] = None,
                 restorer: Optional[Callable[[], Any]] = None,
                 owns_warnings: bool = False,
                 owns_cwd: bool = False) -> None:
        self._writer      = writer
        self._suspender   = suspender
        self._restorer    = restorer
        self.owns_warnings = owns_warnings
        self.owns_cwd      = owns_cwd

    def write(self, text: str) -> None:
        """Raw write to the terminal (escape sequences included)."""
        if self._writer is not None:
            self._writer(text)
        else:
            sys.stdout.write(text)
            sys.stdout.flush()

    @contextmanager
    def suspended(self):
        """Leave the curses session while a subprocess uses the terminal."""
        if self._suspender is not None:
            self._suspender()
            try:
                yield
            finally:
                if self._restorer is not None:
                    self._restorer()
                else:
                    unicurses.doupdate()
        else:
            unicurses.endwin()
            try:
                yield
            finally:
                unicurses.doupdate()

    def clear_screen(self) -> None:
        unicurses.clear()

    # -- warnings ------------------------------------------------------------
    def install_warning_handler(self, handler: Callable) -> Optional[Callable]:
        """Patch warnings.showwarning; returns an opaque restore token."""
        if not self.owns_warnings:
            return None
        previous = warnings.showwarning
        warnings.showwarning = handler
        return previous

    def restore_warning_handler(self, token: Optional[Callable]) -> None:
        if token is not None:
            warnings.showwarning = token

    # -- working directory ---------------------------------------------------
    def set_working_dir(self, path: str) -> None:
        if self.owns_cwd:
            os.chdir(path)


class ProcessTerminalHost(TerminalHost):
    """Standalone CLI behavior: owns the warnings hook and process cwd."""

    def __init__(self, writer: Optional[Callable[[str], None]] = None,
                 suspender: Optional[Callable[[], Any]] = None,
                 restorer: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(writer, suspender, restorer,
                         owns_warnings=True, owns_cwd=True)


class EmbeddedTerminalHost(TerminalHost):
    """Component behavior: never patches warnings and never changes cwd."""

    def __init__(self, writer: Optional[Callable[[str], None]] = None,
                 suspender: Optional[Callable[[], Any]] = None,
                 restorer: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(writer, suspender, restorer,
                         owns_warnings=False, owns_cwd=False)


# ======================== ======================== ========================
#                            SignalBroker
# ======================== ======================== ========================
SignalToken = Tuple[int, Any]


class SignalBroker:
    """Installs signal handlers through an explicit ownership boundary."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def install(self, signum: int, handler: Callable) -> Optional[SignalToken]:
        if not self.enabled:
            return None
        previous = signal.getsignal(signum)
        signal.signal(signum, handler)
        return (signum, previous)

    def restore(self, token: Optional[SignalToken]) -> None:
        if token is None:
            return
        signum, previous = token
        signal.signal(signum, previous)


class ProcessSignalBroker(SignalBroker):
    """Standalone CLI behavior: real process signal installation."""

    def __init__(self) -> None:
        super().__init__(True)


class NullSignalBroker(SignalBroker):
    """Component behavior: install()/restore() are inert no-ops."""

    def __init__(self) -> None:
        super().__init__(False)


# ======================== ======================== ========================
#                            ManagerContext
# ======================== ======================== ========================
@dataclass
class ManagerContext:
    """Instance-owned state of a TUIFIManager.

    Every manager gets its own ManagerContext; two managers in the same
    process never share the copy buffer, markers, ordering, mouse cache or
    base directory.
    """
    base_directory: str = HOME_DIR
    markers: dict = field(default_factory=dict)
    ordered_dirs: dict = field(default_factory=dict)
    order_dirty: bool = False
    copied_files: list = field(default_factory=list)
    copied_from_dir: str = ''
    mouse: MouseEventCache = field(default_factory=MouseEventCache)
    prompt_sink: Optional[Callable[[str, int], None]] = None
