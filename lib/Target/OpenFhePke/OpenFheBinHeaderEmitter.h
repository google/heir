#ifndef LIB_TARGET_OPENFHEBIN_OPENFHEBINHEADEREMITTER_H_
#define LIB_TARGET_OPENFHEBIN_OPENFHEBINHEADEREMITTER_H_

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

void registerToOpenFheBinHeaderTranslation();

/// Translates the given operation to OpenFheBin.
::mlir::LogicalResult translateToOpenFheBinHeader(::mlir::Operation *op,
                                                  llvm::raw_ostream &os);

/// For each function in the mlir module, emits a function header declaration
/// along with any necessary includes.
class OpenFheBinHeaderEmitter {
 public:
  OpenFheBinHeaderEmitter(raw_ostream &os, SelectVariableNames *variableNames);

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
  LogicalResult emitType(Type type, Location loc);
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEBIN_OPENFHEBINHEADEREMITTER_H_
