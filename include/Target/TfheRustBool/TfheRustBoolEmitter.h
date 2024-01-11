#ifndef INCLUDE_TARGET_TFHERUSTBOOL_TFHERUSTBOOLEMITTER_H_
#define INCLUDE_TARGET_TFHERUSTBOOL_TFHERUSTBOOLEMITTER_H_

#include <string>
#include <string_view>

#include "include/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolOps.h"
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
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
::mlir::LogicalResult translateToTfheRustBool(::mlir::Operation *op,
                                              llvm::raw_ostream &os);

class TfheRustBoolEmitter {
 public:
  TfheRustBoolEmitter(raw_ostream &os, SelectVariableNames *variableNames);

  LogicalResult translate(::mlir::Operation &operation);

 private:
  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Pre-populated analysis selecting unique variable names for all the SSA
  /// values.
  SelectVariableNames *variableNames;

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);
  LogicalResult printOperation(::mlir::arith::ConstantOp op);
  LogicalResult printOperation(::mlir::func::FuncOp op);
  LogicalResult printOperation(::mlir::func::ReturnOp op);
  LogicalResult printOperation(CreateTrivialOp op);
  LogicalResult printOperation(tensor::ExtractOp op);
  LogicalResult printOperation(tensor::FromElementsOp op);
  LogicalResult printOperation(AndOp op);
  LogicalResult printOperation(NandOp op);
  LogicalResult printOperation(OrOp op);
  LogicalResult printOperation(NorOp op);
  LogicalResult printOperation(XorOp op);
  LogicalResult printOperation(XnorOp op);

  // Helpers for above
  LogicalResult printSksMethod(::mlir::Value result, ::mlir::Value sks,
                               ::mlir::ValueRange nonSksOperands,
                               std::string_view op,
                               SmallVector<std::string> operandTypes = {});

  // Emit a TfheRustBool type
  LogicalResult emitType(Type type);
  FailureOr<std::string> convertType(Type type);

  void emitAssignPrefix(::mlir::Value result);
};

}  // namespace tfhe_rust_bool
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_TARGET_TFHERUSTBOOL_TFHERUSTBOOLEMITTER_H_
