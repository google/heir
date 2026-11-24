#include "lib/Dialect/CheddarGPU/Transforms/BufferizableOpInterfaceImpl.h"

#include <cassert>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BlockSupport.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"      // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"            // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::heir;
using namespace mlir::heir::cheddar_gpu;

namespace {

// Bufferization model for the destination-style cheddar_gpu ops, which take
// either tensors or buffers. In tensor form an op returns one result per init
// operand; in buffer form it writes to the init operands and has no results.
//
// - Inputs are read. An init is written, and is read only if the op declares
//   `readsDpsInit()`.
// - The ops are elementwise, so an input may share its buffer with an init and
//   the op can bufferize in place.
// - Bufferizing rebuilds the op on buffers with no results. Each result is
//   replaced by the buffer of its tied init.
//
// The ops require all of their polynomial operands to have the same type, so
// buffers that disagree on layout are first cast to a common type.
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
    // `buffers` holds every operand, with each tensor replaced by its buffer.
    SmallVector<Value> buffers;
    SmallVector<unsigned> bufferizedOperands;
    buffers.reserve(op->getNumOperands());
    for (OpOperand& operand : op->getOpOperands()) {
      Value v = operand.get();
      if (!isa<TensorType>(v.getType())) {
        buffers.push_back(v);
        continue;
      }
      FailureOr<Value> buffer = getBuffer(rewriter, v, options, state);
      if (failed(buffer)) return failure();
      buffers.push_back(*buffer);
      bufferizedOperands.push_back(operand.getOperandNumber());
    }

    assert(op->getNumRegions() == 0 && op->getNumSuccessors() == 0);
    rewriter.setInsertionPoint(op);

    // The buffers can disagree on layout, e.g. a fresh allocation next to a
    // function argument or a subview. Any layout can be cast to the fully
    // dynamic one without a copy, so use that as the common operand type.
    SmallVector<Value> newOperands(buffers);
    auto haveSameType = [&]() {
      return llvm::all_of(bufferizedOperands, [&](unsigned i) {
        return newOperands[i].getType() ==
               newOperands[bufferizedOperands.front()].getType();
      });
    };
    if (!haveSameType()) {
      for (unsigned i : bufferizedOperands) {
        auto bufferType = cast<BaseMemRefType>(buffers[i].getType());
        BaseMemRefType dynamicType =
            bufferization::getMemRefTypeWithFullyDynamicLayout(
                cast<TensorType>(op->getOperand(i).getType()),
                bufferType.getMemorySpace());
        if (bufferType == dynamicType) continue;
        FailureOr<Value> dynamicBuffer =
            options.castFn(rewriter, op->getLoc(), dynamicType, buffers[i]);
        if (failed(dynamicBuffer)) return failure();
        newOperands[i] = *dynamicBuffer;
      }
      if (!haveSameType()) {
        return op->emitError()
               << "cannot bufferize: operand buffers are in different memory "
                  "spaces";
      }
    }

    rewriter.insert(Operation::create(
        op->getLoc(), op->getName(), /*resultTypes=*/TypeRange{}, newOperands,
        op->getAttrDictionary(), op->getPropertiesStorage(),
        /*successors=*/BlockRange{}, /*numRegions=*/0));

    // Replace each result with the buffer itself rather than the cast, since
    // that is the buffer type the rest of the bufferization expects.
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    SmallVector<Value> replacements;
    for (OpResult res : op->getResults()) {
      replacements.push_back(
          buffers[dstOp.getTiedOpOperand(res)->getOperandNumber()]);
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
    SubOp::attachInterface<CheddarGPUDpsModel<SubOp>>(*ctx);
    MulOp::attachInterface<CheddarGPUDpsModel<MulOp>>(*ctx);
  });
}
