#ifndef LIB_TARGET_TFHERUST_TFHERUSTHLEMITTER_H_
#define LIB_TARGET_TFHERUST_TFHERUSTHLEMITTER_H_

#include <string>
#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace tfhe_rust {

void registerToTfheRustHLTranslation();

/// Translates the given operation to TfheRustHL.
::mlir::LogicalResult translateToTfheRustHL(::mlir::Operation* op,
                                            llvm::raw_ostream& os);

class TfheRustHLEmitter {
 public:
  TfheRustHLEmitter(raw_ostream& os, SelectVariableNames* variableNames);

  LogicalResult translate(::mlir::Operation& operation);
  bool containsVectorOperands(Operation* op);

 private:
  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Pre-populated analysis selecting unique variable names for all the SSA
  /// values.
  SelectVariableNames* variableNames;

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);
  LogicalResult printOperation(::mlir::func::FuncOp op);
  LogicalResult printOperation(::mlir::func::ReturnOp op);
  LogicalResult printOperation(::mlir::func::CallOp op);
  LogicalResult printOperation(affine::AffineForOp op);
  LogicalResult printOperation(affine::AffineYieldOp op);
  LogicalResult printOperation(affine::AffineStoreOp op);
  LogicalResult printOperation(affine::AffineLoadOp op);
  LogicalResult printOperation(arith::ConstantOp op);
  LogicalResult printOperation(arith::IndexCastOp op);
  LogicalResult printOperation(arith::ShLIOp op);
  LogicalResult printOperation(arith::AndIOp op);
  LogicalResult printOperation(arith::ShRSIOp op);
  LogicalResult printOperation(arith::TruncIOp op);
  LogicalResult printOperation(tensor::ExtractOp op);
  LogicalResult printOperation(tensor::FromElementsOp op);
  LogicalResult printOperation(tensor::InsertOp op);
  LogicalResult printOperation(memref::AllocOp op);
  LogicalResult printOperation(memref::DeallocOp op);
  LogicalResult printOperation(memref::LoadOp op);
  LogicalResult printOperation(memref::GetGlobalOp op);
  LogicalResult printOperation(memref::StoreOp op);
  LogicalResult printOperation(AddOp op);
  LogicalResult printOperation(SubOp op);
  LogicalResult printOperation(MulOp op);
  LogicalResult printOperation(ScalarRightShiftOp op);
  LogicalResult printOperation(CastOp op);
  LogicalResult printOperation(CreateTrivialOp op);
  LogicalResult printOperation(BitAndOp op);
  LogicalResult printOperation(BitOrOp op);
  LogicalResult printOperation(BitXorOp op);

  // Helpers for above
  LogicalResult printMethod(::mlir::Value result,
                            ::mlir::ValueRange nonSksOperands,
                            std::string_view op,
                            SmallVector<std::string> operandTypes = {});
  LogicalResult printBinaryOp(::mlir::Value result, ::mlir::Value lhs,
                              ::mlir::Value rhs, std::string_view op);
  std::string checkOrigin(Value value);

  // Emit a TfheRust type
  LogicalResult emitType(Type type);
  FailureOr<std::string> convertType(Type type);

  void emitAssignPrefix(::mlir::Value result);
  void emitReferenceConversion(::mlir::Value value);
};

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_TFHERUST_TFHERUSTHLEMITTER_H_
