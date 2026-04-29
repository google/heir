#ifndef LIB_DIALECT_KEYMGMT_IR_KEYMGMTDIALECT_TD_
#define LIB_DIALECT_KEYMGMT_IR_KEYMGMTDIALECT_TD_

include "mlir/IR/OpBase.td"

def KeyMgmt_Dialect : Dialect {
  let name = "key_mgmt";
  let summary = "A dialect for managing the lifetime of FHE key material";
  let description = [{
    KeyMgmt is the control dialect for an abstracted key management runtime.
    It individually manages key lifetimes to lower memory utilization in
    FHE programs that utilize a large amount of evaluation keys. It treats
    key material as dataflow elements and provides automatic, fine-grained key
    management.

    The `key_mgmt` dialect provides:

    - New types for keys (e.g., `!key_mgmt.rot_key`).
    - Operations for loading and clearing keys (`key_mgmt.load_key`, `key_mgmt.clear_key`).
    - Prefetching hints to overlap computation with key movement (`key_mgmt.prefetch_key`).

    For more details, see the paper
    [KeyMemRT Compiler and Runtime: Unlocking Memory-Scalable FHE](https://arxiv.org/abs/2601.18445v1)
  }];

  let cppNamespace = "::mlir::heir::key_mgmt";
  let useDefaultTypePrinterParser = 1;
}

#endif // LIB_DIALECT_KEYMGMT_IR_KEYMGMTDIALECT_TD_
