import sys
from pathlib import Path

_mlir_python = __import__("llvm-project.mlir.python", fromlist=["*"])
del sys.modules["llvm-project"]
del sys.modules["llvm-project.mlir"]
del sys.modules["llvm-project.mlir.python"]
for i, p in enumerate(sys.path):
    if "llvm-project" in p:
        del sys.path[i]
sys.path.append(str(Path(_mlir_python.__file__).absolute().parent))

from mlir.ir import *
from mlir.dialects import func
from mlir.dialects import arith
from mlir.dialects import memref
from mlir.dialects import affine
import mlir.extras.types as T


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


@constructAndPrintInModule
def testAffineStoreOp():
    f32 = F32Type.get()
    index_type = IndexType.get()
    memref_type_out = MemRefType.get([12, 12], f32)

    @func.FuncOp.from_py_func(index_type)
    def affine_store_test(arg0):
        mem = memref.AllocOp(memref_type_out, [], []).result

        d0 = AffineDimExpr.get(0)
        s0 = AffineSymbolExpr.get(0)
        map = AffineMap.get(1, 1, [s0 * 3, d0 + s0 + 1])

        # a1 = arith.ConstantOp(f32, 2.1)
        #
        # affine.AffineStoreOp(a1, mem, indices=[arg0, arg0], map=map)

        return mem
