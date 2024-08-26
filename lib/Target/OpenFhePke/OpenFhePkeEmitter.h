#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEPKEEMITTER_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEPKEEMITTER_H_

#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
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
  LogicalResult printOperation(::mlir::arith::ExtSIOp op);
  LogicalResult printOperation(::mlir::arith::IndexCastOp op);
  LogicalResult printOperation(::mlir::func::FuncOp op);
  LogicalResult printOperation(::mlir::func::ReturnOp op);
  LogicalResult printOperation(::mlir::heir::lwe::RLWEEncodeOp op);
  LogicalResult printOperation(::mlir::heir::lwe::RLWEDecodeOp op);
  LogicalResult printOperation(
      ::mlir::heir::lwe::ReinterpretUnderlyingTypeOp op);
  LogicalResult printOperation(AddOp op);
  LogicalResult printOperation(AutomorphOp op);
  LogicalResult printOperation(DecryptOp op);
  LogicalResult printOperation(EncryptOp op);
  LogicalResult printOperation(GenParamsOp op);
  LogicalResult printOperation(GenContextOp op);
  LogicalResult printOperation(GenMulKeyOp op);
  LogicalResult printOperation(GenRotKeyOp op);
  LogicalResult printOperation(KeySwitchOp op);
  LogicalResult printOperation(LevelReduceOp op);
  LogicalResult printOperation(ModReduceOp op);
  LogicalResult printOperation(MulConstOp op);
  LogicalResult printOperation(MulNoRelinOp op);
  LogicalResult printOperation(MulOp op);
  LogicalResult printOperation(MulPlainOp op);
  LogicalResult printOperation(NegateOp op);
  LogicalResult printOperation(RelinOp op);
  LogicalResult printOperation(RotOp op);
  LogicalResult printOperation(SquareOp op);
  LogicalResult printOperation(SubOp op);

  // Helpers for above
  LogicalResult printEvalMethod(::mlir::Value result,
                                ::mlir::Value cryptoContext,
                                ::mlir::ValueRange nonEvalOperands,
                                std::string_view op);

  // Emit an OpenFhe type
  LogicalResult emitType(Type type);

  void emitAutoAssignPrefix(::mlir::Value result);
  LogicalResult emitTypedAssignPrefix(::mlir::Value result);
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEPKEEMITTER_H_
