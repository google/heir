#include "lib/Dialect/Cheddar/Transforms/BufferizableOpInterfaceImpl.h"

#include <cassert>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
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
using namespace mlir::heir::cheddar;

namespace {

// Bufferization model shared by all cheddar DPS ops. The stock DPS model
// supplies the init/result aliasing and write semantics; cheddar refines the
// read side (most ops fully overwrite their destination) and lets an op pass
// the same buffer as input and destination, which every scale-snu kernel
// supports.
template <typename OpTy>
struct CheddarDpsModel
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          CheddarDpsModel<OpTy>, OpTy> {
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

  // Read-write destinations (user interfaces, accumulators) are move-only and
  // cannot be copied, so their init must bufferize in place.
  LogicalResult verifyAnalysis(
      Operation* op, const bufferization::AnalysisState& state) const {
    if (!OpTy::readsDpsInit()) return success();
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    for (OpOperand& init : op->getOpOperands()) {
      if (!dstOp.isDpsInit(&init)) continue;
      if (isa<TensorType>(init.get().getType()) && !state.isInPlace(init))
        return op->emitOpError(
            "move-only read-write destination must bufferize in-place");
    }
    return success();
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
    for (OpResult res : op->getResults())
      replacements.push_back(
          newOperands[dstOp.getTiedOpOperand(res)->getOperandNumber()]);
    bufferization::replaceOpWithBufferizedValues(rewriter, op, replacements);
    return success();
  }
};

template <typename OpTy>
void attachIfDps(MLIRContext* ctx) {
  if constexpr (OpTy::template hasTrait<DestinationStyleOpInterface::Trait>()) {
    OpTy::template attachInterface<CheddarDpsModel<OpTy>>(*ctx);
  }
}

template <typename... OpTys>
void attachAllDpsOps(MLIRContext* ctx) {
  (attachIfDps<OpTys>(ctx), ...);
}

}  // namespace

void mlir::heir::cheddar::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, CheddarDialect* dialect) {
    attachAllDpsOps<
#define GET_OP_LIST
#include "lib/Dialect/Cheddar/IR/CheddarOps.cpp.inc"
        >(ctx);
  });
}
