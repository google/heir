#ifndef LIB_TARGET_SIMFHE_SIMFHEEMITTER_H_
#define LIB_TARGET_SIMFHE_SIMFHEEMITTER_H_

#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Target/SimFHE/SimFHETemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace simfhe {

void registerToSimFHETranslation();

LogicalResult translateToSimFHE(Operation *op, llvm::raw_ostream &os);

class SimFHEEmitter {
 public:
  SimFHEEmitter(llvm::raw_ostream &os, SelectVariableNames *variableNames);
  LogicalResult translate(Operation &operation);

 private:
  raw_indented_ostream os;
  SelectVariableNames *variableNames;

  LogicalResult printOperation(ModuleOp moduleOp);
  LogicalResult printOperation(func::FuncOp funcOp);
  LogicalResult printOperation(func::ReturnOp op);
  LogicalResult printOperation(ckks::AddOp op);
  LogicalResult printOperation(ckks::AddPlainOp op);
  LogicalResult printOperation(ckks::SubOp op);
  LogicalResult printOperation(ckks::SubPlainOp op);
  LogicalResult printOperation(ckks::MulOp op);
  LogicalResult printOperation(ckks::MulPlainOp op);
  LogicalResult printOperation(ckks::NegateOp op);
  LogicalResult printOperation(ckks::RotateOp op);
  LogicalResult printOperation(ckks::RelinearizeOp op);
  LogicalResult printOperation(ckks::RescaleOp op);
  LogicalResult printOperation(ckks::LevelReduceOp op);
  LogicalResult printOperation(ckks::BootstrapOp op);

  std::string getName(Value value) const {
    return variableNames->getNameForValue(value);
  }
};

}  // namespace simfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_SIMFHE_SIMFHEEMITTER_H_
