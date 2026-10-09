#include "lib/Target/TfheRust/Utils.h"

#include <cstdint>
#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallString.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
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
              tensor::EmptyOp, memref::AllocOp, memref::DeallocOp,
              memref::DeallocOp, memref::GetGlobalOp, memref::LoadOp,
              memref::StoreOp, AddOp, SubOp, BitAndOp, BitOrOp, BitXorOp,
              CreateTrivialOp, ApplyLookupTableOp, GenerateLookupTableOp,
              ScalarLeftShiftOp, ScalarRightShiftOp, CastOp, MulOp,
              ::mlir::heir::tfhe_rust_bool::CreateTrivialOp,
              ::mlir::heir::tfhe_rust_bool::AndOp,
              ::mlir::heir::tfhe_rust_bool::PackedOp,
              ::mlir::heir::tfhe_rust_bool::NandOp,
              ::mlir::heir::tfhe_rust_bool::OrOp,
              ::mlir::heir::tfhe_rust_bool::NorOp,
              ::mlir::heir::tfhe_rust_bool::NotOp,
              ::mlir::heir::tfhe_rust_bool::XorOp,
              ::mlir::heir::tfhe_rust_bool::XnorOp,
              ::mlir::heir::preprocessing::LoadResourceOp>(
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

FailureOr<int> getRustIntegerType(int width) {
  for (int candidate : {8, 16, 32, 64, 128}) {
    if (width <= candidate) {
      return candidate;
    }
  }
  return failure();
}

FailureOr<std::string> getRustIntegerTypeStr(IntegerType type) {
  if (type.getWidth() == 1) {
    return std::string("bool");
  }
  auto width = getRustIntegerType(type.getWidth());
  if (failed(width)) return failure();
  return (type.isUnsigned() ? std::string("u") : "") + "i" +
         std::to_string(width.value());
}

FailureOr<DenseElementsAttr> getConstantGlobalData(memref::GetGlobalOp op) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto globalOp =
      dyn_cast<mlir::memref::GlobalOp>(module.lookupSymbol(op.getName()));
  if (!globalOp) {
    return failure();
  }
  auto cstAttr =
      dyn_cast_or_null<DenseElementsAttr>(globalOp.getConstantInitValue());
  if (!cstAttr) {
    return failure();
  }
  return cstAttr;
}

LogicalResult printGetGlobalOp(
    memref::GetGlobalOp op, llvm::raw_ostream& os,
    SelectVariableNames* variableNames,
    llvm::function_ref<FailureOr<std::string>(Type)> convertType) {
  MemRefType memRefType = dyn_cast<MemRefType>(op.getResult().getType());
  if (!memRefType) {
    return op.emitOpError()
           << "Expected global to be a memref " << op.getName();
  }
  auto cstAttr = getConstantGlobalData(op);
  if (failed(cstAttr)) {
    return op.emitOpError() << "Failed to get constant global data";
  }

  auto type = convertType(memRefType.getElementType());
  if (failed(type)) {
    return op.emitOpError()
           << "Failed to emit type for global " << op.getResult().getType();
  }

  // Globals are emitted as 1-D arrays.
  os << "static " << variableNames->getNameForValue(op.getResult())
     << llvm::formatv(" : [{0}; {1}]", type.value(),
                      memRefType.getNumElements())
     << " = [";

  // Populate data by iterating through constant data attribute
  auto printValue = [](const APInt& value) -> std::string {
    llvm::SmallString<40> s;
    value.toStringSigned(s, 10);
    return std::string(s);
  };

  auto cstIter = cstAttr.value().value_begin<APInt>();
  auto cstIterEnd = cstAttr.value().value_end<APInt>();
  os << llvm::join(
      llvm::map_range(llvm::make_range(cstIter, cstIterEnd), printValue), ", ");
  os << "];\n";
  return success();
}

LogicalResult printConstantOp(arith::ConstantOp op, llvm::raw_ostream& os,
                              SelectVariableNames* variableNames) {
  auto valueAttr = op.getValue();
  if (isa<IntegerType>(op.getType()) &&
      op.getType().getIntOrFloatBitWidth() == 1) {
    os << "let " << variableNames->getNameForValue(op.getResult())
       << " : bool = ";
    os << (cast<IntegerAttr>(valueAttr).getValue().isZero() ? "false" : "true")
       << ";\n";
    return success();
  }

  os << "let " << variableNames->getNameForValue(op.getResult()) << " = ";
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    os << intAttr.getValue() << ";\n";
  } else {
    return op.emitError() << "Unknown constant type " << valueAttr.getType();
  }
  return success();
}

LogicalResult printIndexCastOp(
    arith::IndexCastOp op, llvm::raw_ostream& os,
    SelectVariableNames* variableNames,
    llvm::function_ref<LogicalResult(Type)> emitType) {
  os << "let " << variableNames->getNameForValue(op.getOut()) << " = ";
  os << variableNames->getNameForValue(op.getIn()) << " as ";
  if (failed(emitType(op.getOut().getType()))) {
    return op.emitOpError()
           << "Failed to emit index cast type " << op.getOut().getType();
  }
  os << ";\n";
  return success();
}

LogicalResult printTruncIOp(arith::TruncIOp op, llvm::raw_ostream& os,
                            SelectVariableNames* variableNames,
                            llvm::function_ref<LogicalResult(Type)> emitType) {
  os << "let " << variableNames->getNameForValue(op.getResult()) << " = ";
  os << variableNames->getNameForValue(op.getIn());
  if (isa<IntegerType>(op.getType()) &&
      op.getType().getIntOrFloatBitWidth() == 1) {
    // Compare with zero to truncate to a boolean.
    os << " != 0";
  } else {
    os << " as ";
    if (failed(emitType(op.getType()))) {
      return op.emitOpError()
             << "Failed to emit truncated type " << op.getType();
    }
  }
  os << ";\n";
  return success();
}

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir
