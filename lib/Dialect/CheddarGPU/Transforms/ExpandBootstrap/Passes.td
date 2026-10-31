#ifndef LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDBOOTSTRAP_PASSES_TD_
#define LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDBOOTSTRAP_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ExpandBootstrap : Pass<"cheddar-gpu-expand-bootstrap", "ModuleOp"> {
  let summary = "Expand CheddarGPU bootstrap macro operations into leaf operations";

  let description = [{
    Expands bootstrapping composite operations (such as `eval_mod`) into existing
    `cheddar_gpu` operations driven by the compile-time plan in `BootConsts`.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::gpu::GPUDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::heir::cheddar_gpu::CheddarGPUDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDBOOTSTRAP_PASSES_TD_
