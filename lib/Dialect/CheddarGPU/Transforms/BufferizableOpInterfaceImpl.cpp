#include "lib/Dialect/CheddarGPU/Transforms/BufferizableOpInterfaceImpl.h"

#include <cassert>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::heir;
using namespace mlir::heir::cheddar_gpu;

namespace {

template <typename OpTy>
struct CheddarGPUDpsModel
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          CheddarGPUDpsModel<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    return !dstOp.isDpsInit(&opOperand) || OpTy::readsDpsInit();
  }

  // An op may read and write the same (equivalent) buffer without a conflict.
  bool bufferizesToElementwiseAccess(Operation* op,
                                     const bufferization::AnalysisState& state,
                                     ArrayRef<OpOperand*> opOperands) const {
    return true;
  }

  // Rebuild the op on buffers with no results; each result is replaced by the
  // buffer of its tied init.
  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const bufferization::BufferizationOptions& options,
                          bufferization::BufferizationState& state) const {
    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands());
    for (OpOperand& operand : op->getOpOperands()) {
      Value v = operand.get();
      if (!isa<TensorType>(v.getType())) {
        newOperands.push_back(v);
        continue;
      }
      FailureOr<Value> buffer = getBuffer(rewriter, v, options, state);
      if (failed(buffer)) return failure();
      newOperands.push_back(*buffer);
    }

    assert(op->getNumRegions() == 0 && op->getNumSuccessors() == 0);
    rewriter.setInsertionPoint(op);
    rewriter.insert(Operation::create(
        op->getLoc(), op->getName(), /*resultTypes=*/TypeRange{}, newOperands,
        op->getAttrDictionary(), op->getPropertiesStorage(),
        /*successors=*/BlockRange{}, /*numRegions=*/0));

    auto dstOp = cast<DestinationStyleOpInterface>(op);
    SmallVector<Value> replacements;
    for (OpResult res : op->getResults()) {
      replacements.push_back(
          newOperands[dstOp.getTiedOpOperand(res)->getOperandNumber()]);
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, replacements);
    return success();
  }
};

}  // namespace

void mlir::heir::cheddar_gpu::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, CheddarGPUDialect* dialect) {
    AddOp::attachInterface<CheddarGPUDpsModel<AddOp>>(*ctx);
  });
}
