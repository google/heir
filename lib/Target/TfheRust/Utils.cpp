#include "lib/Target/TfheRust/Utils.h"

#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace tfhe_rust {

LogicalResult canEmitFuncForTfheRust(func::FuncOp &funcOp) {
  WalkResult failIfInterrupted = funcOp.walk([&](Operation *op) {
    return TypeSwitch<Operation *, WalkResult>(op)
        // This list should match the list of implemented overloads of
        // `printOperation`.
        .Case<ModuleOp, func::FuncOp, func::ReturnOp, affine::AffineForOp,
              affine::AffineYieldOp, arith::ConstantOp, arith::IndexCastOp,
              arith::ShLIOp, arith::AndIOp, arith::ShRSIOp, arith::TruncIOp,
              tensor::ExtractOp, tensor::FromElementsOp, memref::AllocOp,
              memref::DeallocOp, memref::GetGlobalOp, memref::LoadOp,
              memref::StoreOp, AddOp, BitAndOp, CreateTrivialOp,
              ApplyLookupTableOp, GenerateLookupTableOp, ScalarLeftShiftOp,
              ::mlir::heir::tfhe_rust_bool::CreateTrivialOp,
              ::mlir::heir::tfhe_rust_bool::AndOp,
              ::mlir::heir::tfhe_rust_bool::NandOp,
              ::mlir::heir::tfhe_rust_bool::OrOp,
              ::mlir::heir::tfhe_rust_bool::NorOp,
              ::mlir::heir::tfhe_rust_bool::NotOp,
              ::mlir::heir::tfhe_rust_bool::XorOp,
              ::mlir::heir::tfhe_rust_bool::XnorOp>(
            [&](auto op) { return WalkResult::advance(); })
        .Default([&](Operation *op) {
          llvm::errs()
              << "Skipping function " << funcOp.getName()
              << " which cannot be emitted because it has an unsupported op: "
              << *op << "\n";
          return WalkResult::interrupt();
        });
  });

  if (failIfInterrupted.wasInterrupted()) return failure();
  return success();
}

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir
