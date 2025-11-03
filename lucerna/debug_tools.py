"""
Tools to perform actions and checks only when Python debugging is enabled (the variable __debug__ is true).

To disable debugging, run Python with the -O (optimize) flag.
"""

from typing import NoReturn
import sys


def debug_raise() -> None | NoReturn:
    """Reraise current exception."""
    is_in_exception_handler = sys.exc_info()[0] is not None

    if is_in_exception_handler and __debug__:
        raise  # pylint: disable=misplaced-bare-raise
