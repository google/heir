#ifndef INCLUDE_TARGET_OPENFHEPKE_OPENFHEPKEEMITTER_H_
#define INCLUDE_TARGET_OPENFHEPKE_OPENFHEPKEEMITTER_H_

#include <string_view>

#include "include/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "include/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

void registerToOpenFhePkeTranslation();

/// Translates the given operation to OpenFhePke.
::mlir::LogicalResult translateToOpenFhePke(::mlir::Operation *op,
                                            llvm::raw_ostream &os);

class OpenFhePkeEmitter {
 public:
  OpenFhePkeEmitter(raw_ostream &os, SelectVariableNames *variableNames);

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
  LogicalResult printOperation(AddOp op);
  LogicalResult printOperation(SubOp op);
  LogicalResult printOperation(MulOp op);
  LogicalResult printOperation(MulPlainOp op);
  LogicalResult printOperation(MulConstOp op);
  LogicalResult printOperation(NegateOp op);
  LogicalResult printOperation(SquareOp op);
  LogicalResult printOperation(RelinOp op);
  LogicalResult printOperation(ModReduceOp op);
  LogicalResult printOperation(LevelReduceOp op);
  LogicalResult printOperation(RotOp op);
  LogicalResult printOperation(AutomorphOp op);
  LogicalResult printOperation(KeySwitchOp op);

  // Helpers for above
  LogicalResult printEvalMethod(::mlir::Value result,
                                ::mlir::Value cryptoContext,
                                ::mlir::ValueRange nonEvalOperands,
                                std::string_view op);

  // Emit an OpenFhe type
  LogicalResult emitType(Type type);

  void emitAssignPrefix(::mlir::Value result);
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_TARGET_OPENFHEPKE_OPENFHEPKEEMITTER_H_
