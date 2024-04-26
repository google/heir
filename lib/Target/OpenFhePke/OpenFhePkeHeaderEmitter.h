#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEPKEHEADEREMITTER_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEPKEHEADEREMITTER_H_

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

void registerToOpenFhePkeHeaderTranslation();

/// Translates the given operation to OpenFhePke.
::mlir::LogicalResult translateToOpenFhePkeHeader(::mlir::Operation *op,
                                                  llvm::raw_ostream &os);

/// For each function in the mlir module, emits a function header declaration
/// along with any necessary includes.
class OpenFhePkeHeaderEmitter {
 public:
  OpenFhePkeHeaderEmitter(raw_ostream &os, SelectVariableNames *variableNames);

  LogicalResult translate(::mlir::Operation &operation);

 private:
  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Pre-populated analysis selecting unique variable names for all the SSA
  /// values.
  SelectVariableNames *variableNames;

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);
  LogicalResult printOperation(::mlir::func::FuncOp op);

  // Emit an OpenFhe type
  LogicalResult emitType(Type type);
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEPKEHEADEREMITTER_H_
