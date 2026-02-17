#ifndef LIB_DIALECT_KEYMEMRUNTIME_IR_KEYMEMRUNTIMEDIALECT_TD_
#define LIB_DIALECT_KEYMEMRUNTIME_IR_KEYMEMRUNTIMEDIALECT_TD_

include "mlir/IR/OpBase.td"

def KeyMemRuntime_Dialect : Dialect {
  let name = "kmrt";
  let summary = "A dialect for KeyMemRuntime operations";
  let description = [{
    KeyMemRuntime individually manages rotation key lifetimes to lower memory
    utilization and allow an arbitrary number of rotation indices to be
    supported without memory bloating. It treats rotation keys as dataflow
    elements and provides automatic, fine-grained key management.

    The KeyMemRuntime dialect provides:

    - A new type for rotation keys (`!kmrt.rot_key`).
    - Operations for loading and clearing keys (`kmrt.load_key`, `kmrt.clear_key`).
    - Prefetching hints to overlap computation with key movement (`kmrt.prefetch_key`).
    - Support for rotation decomposition and bootstrap key management.

    For more details, see the paper
    [KeyMemRT Compiler and Runtime: Unlocking Memory-Scalable FHE](https://arxiv.org/abs/2601.18445v1)
  }];

  let cppNamespace = "::mlir::heir::kmrt";
  let useDefaultTypePrinterParser = 1;
}

#endif // LIB_DIALECT_KEYMEMRUNTIME_IR_KEYMEMRUNTIMEDIALECT_TD_
