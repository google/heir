#include "lib/Dialect/ReducesLevelOpInterfaceRegistration.h"

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

namespace {

template <typename OpTy>
struct LinalgReducesLevelOpInterfaceModel
    : public ReducesLevelOpInterface::ExternalModel<
          LinalgReducesLevelOpInterfaceModel<OpTy>, OpTy> {
  int getLevelsToDrop(Operation* op) const { return 1; }

  SmallVector<OpOperand*> getOperandsToReduce(
      Operation* op, const DataFlowSolver* solver) const {
    SmallVector<OpOperand*> result;
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    for (OpOperand* operand : dpsOp.getDpsInputOperands()) {
      if (!solver || isSecret(operand->get(), solver)) {
        result.push_back(operand);
      }
    }
    if (result.empty() && !dpsOp.getDpsInputOperands().empty()) {
      result.push_back(dpsOp.getDpsInputOperands()[0]);
    }
    return result;
  }
};

}  // namespace

void registerReducesLevelOpInterfaceExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, linalg::LinalgDialect* dialect) {
    linalg::MatmulOp::attachInterface<
        LinalgReducesLevelOpInterfaceModel<linalg::MatmulOp>>(*ctx);
    linalg::MatvecOp::attachInterface<
        LinalgReducesLevelOpInterfaceModel<linalg::MatvecOp>>(*ctx);
    linalg::Conv1DNcwFcwOp::attachInterface<
        LinalgReducesLevelOpInterfaceModel<linalg::Conv1DNcwFcwOp>>(*ctx);
    linalg::Conv2DNchwFchwOp::attachInterface<
        LinalgReducesLevelOpInterfaceModel<linalg::Conv2DNchwFchwOp>>(*ctx);
  });
}

}  // namespace heir
}  // namespace mlir
