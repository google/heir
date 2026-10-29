#ifndef LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDHOIST_PASSES_TD_
#define LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDHOIST_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ExpandHoist : Pass<"cheddar-gpu-expand-hoist", "ModuleOp"> {
  let summary = "Expand CheddarGPU hoisted linear transform phase into leaf operations";

  let description = [{
    Expands `cheddar_gpu.hoist_phase` operations into fused baby-step and giant-step
    operations and reduction loops driven by the compile-time plan in `BootConsts`.
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

#endif  // LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDHOIST_PASSES_TD_
