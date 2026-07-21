#ifndef LIB_TARGET_POULPY_POULPYEMITTER_H_
#define LIB_TARGET_POULPY_POULPYEMITTER_H_

#include <string>
#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace poulpy {
void registerToPoulpyTranslation();

/// Translates the given operation to Poulpy.
::mlir::LogicalResult translateToPoulpy(::mlir::Operation* op,
                                        llvm::raw_ostream& os);

class PoulpyEmitter {
 public:
  PoulpyEmitter(raw_ostream& os, SelectVariableNames* variableNames);

  LogicalResult translate(::mlir::Operation& operation);
  LogicalResult translateBlock(::mlir::Block& block);

 private:
  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Pre-populated analysis selecting unique variable names for all the SSA
  /// values.
  SelectVariableNames* variableNames;

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);

  // Emit a Poulpy type
  LogicalResult emitType(Type type);
  FailureOr<std::string> convertType(Type type);
};

}  // namespace poulpy
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_POULPY_POULPYEMITTER_H_