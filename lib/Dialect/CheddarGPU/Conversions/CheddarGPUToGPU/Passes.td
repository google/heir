#ifndef LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_PASSES_TD_
#define LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_PASSES_TD_

include "mlir/Pass/PassBase.td"

def CheddarGPUToGPU : Pass<"cheddar-gpu-to-gpu", "ModuleOp"> {
  let summary = "Lower `cheddar_gpu` operations to `gpu.launch_func` kernels.";

  let description = [{
    This pass lowers operations in the `cheddar_gpu` dialect to 1-D GPU kernels.

    (* example filepath=tests/Dialect/CheddarGPU/Conversions/doctest.mlir *)
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::DLTIDialect",
    "mlir::gpu::GPUDialect",
    "mlir::memref::MemRefDialect",
    "mlir::heir::cheddar_gpu::CheddarGPUDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_PASSES_TD_
