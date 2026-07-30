#ifndef LIB_TARGET_POULPY_POULPYEMITTER_H_
#define LIB_TARGET_POULPY_POULPYEMITTER_H_

#include <string>
#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Poulpy/IR/PoulpyOps.h"
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
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

  llvm::DenseSet<Value> mutatedValues;
  llvm::DenseSet<Value> pendingAllocs;

  void computeMutatedValues(func::FuncOp funcOp);
  void materializeIfPending(Value dst, Value module, Value layoutSource);
  LogicalResult checkNotPending(Value dst, Operation* op);

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);
  LogicalResult printOperation(func::FuncOp op);
  LogicalResult printOperation(func::ReturnOp op);
  LogicalResult printOperation(memref::AllocOp op);
  LogicalResult printOperation(AddOp op);
  LogicalResult printOperation(AddAssignOp op);
  LogicalResult printOperation(SubOp op);
  LogicalResult printOperation(SubAssignOp op);
  LogicalResult printOperation(MulOp op);
  LogicalResult printOperation(MulAssignOp op);

  // Emit a Poulpy type
  LogicalResult emitType(Type type, bool isArg, bool isMutated);
  FailureOr<std::string> convertType(Type type, bool isArg, bool isMutated);
};

}  // namespace poulpy
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_POULPY_POULPYEMITTER_H_
