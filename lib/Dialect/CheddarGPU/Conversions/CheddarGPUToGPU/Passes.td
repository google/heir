#ifndef LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_PASSES_TD_
#define LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_PASSES_TD_

include "mlir/Pass/PassBase.td"

def CheddarGPUToGPU : Pass<"cheddar-gpu-to-gpu", "ModuleOp"> {
  let summary = "Lower `cheddar_gpu` element-wise and permute operations to `gpu.launch` kernels.";

  let description = [{
    This pass lowers element-wise, permute, and accumulation operations in the
    `cheddar_gpu` dialect to 1-D `gpu.launch` kernels and modular arithmetic (`mod_arith`).
  }];

  let options = [
    Option<"nttImpl", "ntt-impl", "std::string", /*default=*/"\"simple\"",
           "NTT implementation algorithm ('simple' or 'fused')">,
  ];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::DLTIDialect",
    "mlir::func::FuncDialect",
    "mlir::gpu::GPUDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::heir::cheddar_gpu::CheddarGPUDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_PASSES_TD_
