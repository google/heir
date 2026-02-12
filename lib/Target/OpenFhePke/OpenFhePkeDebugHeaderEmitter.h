#ifndef LIB_TARGET_OPENFHEPKE_DEBUGHELPEREMITTER_H_
#define LIB_TARGET_OPENFHEPKE_DEBUGHELPEREMITTER_H_

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {
 

::mlir::LogicalResult translateToOpenFhePkeDebugHeaderEmitter(
    ::mlir::Operation* op, 
    llvm::raw_ostream& os, 
    OpenfheImportType importType);

/// For each function in the mlir module, emits a function header declaration
/// along with any necessary includes.
class OpenFhePkeDebugHeaderEmitter{
 public:
  OpenFhePkeDebugHeaderEmitter(raw_ostream& os, SelectVariableNames* variableNames,
                    OpenfheImportType importType);

  LogicalResult translate(::mlir::Operation& operation);
  
 private:                    
  OpenfheImportType importType_;
  
  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Pre-populated analysis selecting unique variable names for all the SSA
  /// values.
  SelectVariableNames* variableNames;

  bool isEmitted;

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);
  LogicalResult printOperation(::mlir::func::FuncOp op);

  // Emit an OpenFhe type
  LogicalResult emitType(::mlir::Type type, ::mlir::Location loc);
};
  
}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_DEBUGHELPEREMITTER_H_