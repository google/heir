#ifndef LIB_TARGET_TFHERUSTBOOL_TFHERUSTBOOLEMITTER_H_
#define LIB_TARGET_TFHERUSTBOOL_TFHERUSTBOOLEMITTER_H_

#include <string>
#include <string_view>
#include <utility>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOps.h"
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
namespace tfhe_rust_bool {

void registerToTfheRustBoolTranslation();

/// Translates the given operation to TfheRustBool.
::mlir::LogicalResult translateToTfheRustBool(::mlir::Operation* op,
                                              llvm::raw_ostream& os,
                                              bool packedAPI);

class TfheRustBoolEmitter {
 public:
  TfheRustBoolEmitter(raw_ostream& os, SelectVariableNames* variableNames,
                      bool packedAPI);

  LogicalResult translate(::mlir::Operation& operation);
  bool containsVectorOperands(Operation* op);

 private:
  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Pre-populated analysis selecting unique variable names for all the SSA
  /// values.
  SelectVariableNames* variableNames;

  // Boolean to keep track if the packed API is used or not
  bool packedAPI;

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);
  LogicalResult printOperation(::mlir::func::FuncOp op);
  LogicalResult printOperation(::mlir::func::ReturnOp op);
  LogicalResult printOperation(CreateTrivialOp op);
  LogicalResult printOperation(affine::AffineForOp op);
  LogicalResult printOperation(affine::AffineYieldOp op);
  LogicalResult printOperation(arith::ConstantOp op);
  LogicalResult printOperation(arith::IndexCastOp op);
  LogicalResult printOperation(arith::ShLIOp op);
  LogicalResult printOperation(arith::AndIOp op);
  LogicalResult printOperation(arith::ShRSIOp op);
  LogicalResult printOperation(arith::TruncIOp op);
  LogicalResult printOperation(tensor::ExtractOp op);
  LogicalResult printOperation(tensor::FromElementsOp op);
  LogicalResult printOperation(memref::AllocOp op);
  LogicalResult printOperation(memref::LoadOp op);
  LogicalResult printOperation(memref::StoreOp op);
  LogicalResult printOperation(AndOp op);
  LogicalResult printOperation(NandOp op);
  LogicalResult printOperation(OrOp op);
  LogicalResult printOperation(NorOp op);
  LogicalResult printOperation(NotOp op);
  LogicalResult printOperation(XorOp op);
  LogicalResult printOperation(XnorOp op);
  LogicalResult printOperation(PackedOp op);

  // Helpers for above
  LogicalResult printSksMethod(::mlir::Value result, ::mlir::Value sks,
                               ::mlir::ValueRange nonSksOperands,
                               std::string_view op,
                               SmallVector<std::string> operandTypes = {});
  LogicalResult printBinaryOp(::mlir::Value result, ::mlir::Value lhs,
                              ::mlir::Value rhs, std::string_view op);
  std::pair<std::string, std::string> checkOrigin(Value value);

  // Emit a TfheRustBool type
  LogicalResult emitType(Type type);
  FailureOr<std::string> convertType(Type type);

  void emitAssignPrefix(::mlir::Value result);
  void emitReferenceConversion(::mlir::Value value);
};

}  // namespace tfhe_rust_bool
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_TFHERUSTBOOL_TFHERUSTBOOLEMITTER_H_
