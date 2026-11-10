"""Registers HEIR's dialects and passes with the MLIR Python bindings.

The upstream `_mlir_libs/__init__.py` imports `_site_initialize_<i>` modules
automatically, and calls their `register_dialects` function on the registry
that is appended to every new `Context`.
"""

from . import _heir
from ._heir import register_dialects  # noqa: F401

_heir.register_passes()
