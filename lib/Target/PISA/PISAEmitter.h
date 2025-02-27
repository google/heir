#ifndef LIB_TARGET_PISA_PISAEMITTER_H_
#define LIB_TARGET_PISA_PISAEMITTER_H_

#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/PISA/IR/PISAOps.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace pisa {

void registerToPISATranslation();
void registerToPISAInputsTranslation();

/// Translates the given operation to PISA
LogicalResult translateToPISA(Operation *op, llvm::raw_ostream &os,
                              bool emitInputOnly);

class PISAEmitter {
 public:
  PISAEmitter(raw_ostream &os, SelectVariableNames *variableNames,
              bool emitInputOnly);

  LogicalResult translate(::mlir::Operation &operation);

 private:
  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Whether to only output input/output file or instruction stream
  bool emitInputOnly;

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
  LogicalResult printOperation(MuliOp op);
  LogicalResult printOperation(MacOp op);
  LogicalResult printOperation(MaciOp op);
  LogicalResult printOperation(NTTOp op);
  LogicalResult printOperation(INTTOp op);

  // Helpers for above
  LogicalResult printPISAOp(std::string_view name, Value result,
                            ValueRange operands, int index = -1,
                            StringRef immediate = "");
};

}  // namespace pisa
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_PISA_PISAEMITTER_H_
