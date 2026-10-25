#include "lib/Dialect/Poulpy/Transforms/BufferizableOpInterfaceImpl.h"

#include <cassert>

#include "lib/Dialect/Poulpy/IR/PoulpyDialect.h"
#include "lib/Dialect/Poulpy/IR/PoulpyOps.h"
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
using namespace mlir::heir::poulpy;

namespace {

// Bufferization model shared by all poulpy DPS ops. The stock DPS model
// supplies the init/result aliasing and write semantics; poulpy refines the
// read side, since only the _assign ops read their destination. A destination
// never shares a buffer with another operand: the Rust API cannot borrow one
// value both mutably and immutably, so in-place updates go through the
// _assign ops instead.
template <typename OpTy>
struct PoulpyDpsModel
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          PoulpyDpsModel<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    return !dstOp.isDpsInit(&opOperand) || OpTy::readsDpsInit();
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
    OpTy::template attachInterface<PoulpyDpsModel<OpTy>>(*ctx);
  }
}

template <typename... OpTys>
void attachAllDpsOps(MLIRContext* ctx) {
  (attachIfDps<OpTys>(ctx), ...);
}

}  // namespace

void mlir::heir::poulpy::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, PoulpyDialect* dialect) {
    attachAllDpsOps<
#define GET_OP_LIST
#include "lib/Dialect/Poulpy/IR/PoulpyOps.cpp.inc"
        >(ctx);
  });
}
