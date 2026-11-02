#ifndef LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOSCF_PASSES_TD_
#define LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOSCF_PASSES_TD_

include "mlir/Pass/PassBase.td"

def CheddarGPUToSCF : Pass<"cheddar-gpu-to-scf", "ModuleOp"> {
  let summary = "Lower `cheddar_gpu` operations to host reference loops and `mod_arith`.";

  let description = [{
    This pass lowers element-wise, permute, and accumulation operations in the
    `cheddar_gpu` dialect to `scf.for` loops and modular arithmetic (`mod_arith`).
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::heir::cheddar_gpu::CheddarGPUDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOSCF_PASSES_TD_
