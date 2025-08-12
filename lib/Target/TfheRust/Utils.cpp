#include "lib/Target/TfheRust/Utils.h"

#include <cstdint>

#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"       // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace tfhe_rust {

// TODO: Fix this function to match the list of implemented ops
LogicalResult canEmitFuncForTfheRust(func::FuncOp& funcOp) {
  WalkResult failIfInterrupted = funcOp.walk([&](Operation* op) {
    return TypeSwitch<Operation*, WalkResult>(op)
        // This list should match the list of implemented overloads of
        // `printOperation`.
        .Case<ModuleOp, func::FuncOp, func::ReturnOp, func::CallOp,
              affine::AffineForOp, affine::AffineYieldOp, affine::AffineLoadOp,
              affine::AffineStoreOp, arith::ConstantOp, arith::IndexCastOp,
              arith::ShLIOp, arith::AndIOp, arith::ShRSIOp, arith::TruncIOp,
              tensor::ExtractOp, tensor::FromElementsOp, tensor::InsertOp,
              memref::AllocOp, memref::DeallocOp, memref::DeallocOp,
              memref::GetGlobalOp, memref::LoadOp, memref::StoreOp, AddOp,
              SubOp, BitAndOp, BitOrOp, BitXorOp, CreateTrivialOp,
              ApplyLookupTableOp, GenerateLookupTableOp, ScalarLeftShiftOp,
              ScalarRightShiftOp, CastOp, MulOp,
              ::mlir::heir::tfhe_rust_bool::CreateTrivialOp,
              ::mlir::heir::tfhe_rust_bool::AndOp,
              ::mlir::heir::tfhe_rust_bool::PackedOp,
              ::mlir::heir::tfhe_rust_bool::NandOp,
              ::mlir::heir::tfhe_rust_bool::OrOp,
              ::mlir::heir::tfhe_rust_bool::NorOp,
              ::mlir::heir::tfhe_rust_bool::NotOp,
              ::mlir::heir::tfhe_rust_bool::XorOp,
              ::mlir::heir::tfhe_rust_bool::XnorOp>(
            [&](auto op) { return WalkResult::advance(); })
        .Default([&](Operation* op) {
          llvm::errs()
              << "Skipping function " << funcOp.getName()
              << " which cannot be emitted because it has an unsupported op: "
              << *op << "\n"
              << "Origin: TfheRust/Utils.cpp:canEmitFuncForTfheRust\n";
          return WalkResult::interrupt();
        });
  });

  if (failIfInterrupted.wasInterrupted()) return failure();
  return success();
}

int16_t getTfheRustBitWidth(Type type) {
  if (isa<tfhe_rust::EncryptedUInt2Type>(type)) {
    return 2;
  }
  if (isa<tfhe_rust::EncryptedUInt3Type>(type)) {
    return 3;
  }
  if (isa<tfhe_rust::EncryptedUInt4Type>(type)) {
    return 4;
  }
  if (isa<tfhe_rust::EncryptedUInt8Type>(type) ||
      isa<tfhe_rust::EncryptedInt8Type>(type)) {
    return 8;
  }
  if (isa<tfhe_rust::EncryptedUInt10Type>(type)) {
    return 10;
  }
  if (isa<tfhe_rust::EncryptedUInt12Type>(type)) {
    return 12;
  }
  if (isa<tfhe_rust::EncryptedUInt14Type>(type)) {
    return 14;
  }
  if (isa<tfhe_rust::EncryptedUInt16Type>(type) ||
      isa<tfhe_rust::EncryptedInt16Type>(type)) {
    return 16;
  }
  if (isa<tfhe_rust::EncryptedUInt32Type>(type) ||
      isa<tfhe_rust::EncryptedInt32Type>(type)) {
    return 32;
  }
  if (isa<tfhe_rust::EncryptedUInt64Type>(type) ||
      isa<tfhe_rust::EncryptedInt64Type>(type)) {
    return 64;
  }
  if (isa<tfhe_rust::EncryptedUInt128Type>(type) ||
      isa<tfhe_rust::EncryptedInt128Type>(type)) {
    return 128;
  }
  if (isa<tfhe_rust::EncryptedUInt256Type>(type) ||
      isa<tfhe_rust::EncryptedInt256Type>(type)) {
    return 256;
  }
  return -1;
}

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir
