from heir.python._mlir.ir import *

print(dir())

from heir.python._mlir.dialects import func
from heir.python._mlir.dialects import arith
from heir.python._mlir.dialects import memref
from heir.python._mlir.dialects import affine
import heir.python._mlir.extras.types as T


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
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

        a1 = arith.ConstantOp(f32, 2.1)

        affine.AffineStoreOp(a1, mem, indices=[arg0, arg0], map=map)

        return mem
