#ifndef LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDMODSWITCH_PASSES_TD_
#define LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDMODSWITCH_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ExpandModSwitch : Pass<"cheddar-gpu-expand-modswitch", "ModuleOp"> {
  let summary = "Expand CheddarGPU ModSwitch macro operations into leaf operations and tables";

  let description = [{
    Expands macro operations (`pseudo_mod_up`, `mod_up`, `mod_down`, `rescale`,
    `mod_down_rescale`) into leaf operations (`intt`, `bconv`, `ntt`, `mul_const`,
    `mod_down_epilogue`, `memref.copy`) and compiles required constant tables into
    `memref.global` constants.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::heir::cheddar_gpu::CheddarGPUDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDMODSWITCH_PASSES_TD_
