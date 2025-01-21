#ifndef LIB_TARGET_OPTALYSYS_OPTALYSYSEMITTER_H_
#define LIB_TARGET_OPTALYSYS_OPTALYSYSEMITTER_H_

#include <string>
#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Optalysys/IR/OptalysysOps.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace optalysys {

/// Translates the given operation to Optalysys
::mlir::LogicalResult translateToOptalysys(::mlir::Operation *op,
                                           llvm::raw_ostream &os);

class OptalysysEmitter {
 public:
  OptalysysEmitter(raw_ostream &os, SelectVariableNames *variableNames);

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
  LogicalResult printOperation(::mlir::func::ReturnOp op);
  LogicalResult printOperation(::mlir::arith::ConstantOp op);

  // Optalysys ops
  LogicalResult printOperation(RlweGenLutOp op);
  LogicalResult printOperation(DeviceMultiCMux1024PbsBOp op);
  LogicalResult printOperation(TrivialLweEncryptOp op);
  LogicalResult printOperation(MulScalarOp op);
  LogicalResult printOperation(AddOp op);
  LogicalResult printOperation(LweCreateBatchOp op);
  LogicalResult printOperation(LoadOp op);

  // helper on name and type
  std::string getName(::mlir::Value value) {
    return variableNames->getNameForValue(value);
  }

  // Emit an Optalysys type
  LogicalResult emitType(Type type, ::mlir::Location loc);

  LogicalResult emitTypedAssignPrefix(::mlir::Value result,
                                      ::mlir::Location loc);
};

void registerToOptalysysTranslation(void);

}  // namespace optalysys
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPTALYSYS_OPTALYSYSEMITTER_H_
