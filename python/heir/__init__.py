"""HEIR: an MLIR-based compiler for fully homomorphic encryption.

This package contains Python bindings for HEIR's MLIR dialects, passes, and
code emitters, plus a thin layer of convenience functions on top of them.

Example:

  import heir

  with heir.context():
    module = heir.parse(open("input.mlir").read())
    heir.run(module, "mlir-to-bgv", heir.Pass("scheme-to-openfhe"))
    header = heir.translate(module, "emit-openfhe-pke-header")
    source = heir.translate(module, "emit-openfhe-pke")

The MLIR Python API lives in the usual places under this package, e.g.
`heir.ir`, `heir.passmanager`, and `heir.dialects.<dialect>` for both upstream
MLIR dialects (`arith`, `func`, `tensor`, ...) and HEIR dialects (`secret`,
`bgv`, `ckks`, `lwe`, `openfhe`, ...).
"""

from . import ir
from . import passmanager
from . import types
from ._context import context
from ._context import module
from ._context import parse
from ._context import parse_file
from ._passes import Pass
from ._passes import pipeline_string
from ._passes import run
from ._translate import TranslationError
from ._translate import translate
from ._translate import translations

__all__ = [
    "Pass",
    "TranslationError",
    "context",
    "ir",
    "module",
    "parse",
    "parse_file",
    "passmanager",
    "pipeline_string",
    "run",
    "translate",
    "translations",
    "types",
]
