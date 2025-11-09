"""
Tools to perform actions and checks only when Python debugging is enabled (the variable __debug__ is true).

To disable debugging, run Python with the -O (optimize) flag.
"""

from typing import NoReturn
import sys


def is_exception_active() -> bool:
    """Check if currently in an exception handler."""
    is_in_exception_handler = sys.exc_info()[0] is not None

    return is_in_exception_handler


def is_debug() -> bool:
    """Check if Python debugging is enabled."""
    return __debug__


def debug_reraise() -> None | NoReturn:
    """Reraise current exception."""
    if is_debug() and is_exception_active():
        raise  # pylint: disable=misplaced-bare-raise


def debug_raise(exception: BaseException) -> None | NoReturn:
    """Raise given exception."""
    if is_debug():
        raise  exception
