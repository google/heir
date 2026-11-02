#ifndef LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDKEYSWITCH_PASSES_TD_
#define LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDKEYSWITCH_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ExpandKeySwitch : Pass<"cheddar-gpu-expand-keyswitch", "ModuleOp"> {
  let summary = "Expand CheddarGPU KeySwitch macro operations into leaf operations and tables";

  let description = [{
    Expands key switching macro operations (`key_switch_no_mod_down`, `key_switch`,
    `relinearize`, `relinearize_rescale`, `hrot`, `hrot_add`, `hconj`) into leaf
    operations (`mod_up`, `cp_accum`, `cp_accum_add`, `mul_const`, `mod_down`,
    `mod_down_rescale`, `permute`, `permute_accum`, `add`).
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::gpu::GPUDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::heir::cheddar_gpu::CheddarGPUDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDKEYSWITCH_PASSES_TD_
