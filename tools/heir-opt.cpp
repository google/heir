#include "include/Dialect/EncryptedArith/IR/EncryptedArithDialect.h"
#include "include/Dialect/HEIR/IR/HEIRDialect.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h" // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h" // from @llvm-project

// TODO(b/281566825): Add a memref2arith lowering pipeline that chains:
//   1. fold-memref-alias-ops: Remove expand, subview, and collapse
//   2. [custom] lower-copy: Lower memref copies to affine stores and
//   loads.
//   3. [custom] forward-global: Forward global memref accesses with their
//   constant values.
//   4. affine-scalrep: Forward stores to loads and remove redundant
//   loads.
//   5. MemrefAllocRemovalPattern: Removes unused memref allocations.

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::heir::HEIRDialect>();
  registry.insert<mlir::heir::EncryptedArithDialect>();

  // Add expected MLIR dialects to the registry.
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  return failed(mlir::MlirOptMain(argc, argv, "HEIR Pass Driver", registry));
}
