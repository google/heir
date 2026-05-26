#include "lib/Dialect/Lattigo/Transforms/BufferizableOpInterfaceImpl.h"

#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::heir;
using namespace mlir::heir::lattigo;

namespace {

template <typename OpTy>
struct DecodeOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          DecodeOpInterface<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    return true;
  };

  bool bufferizesToMemoryWrite(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    return opOperand.getOperandNumber() == 2;  // $value
  };

  bufferization::AliasingValueList getAliasingValues(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    if (opOperand.getOperandNumber() == 2) {
      return {{op->getResult(0), bufferization::BufferRelation::Equivalent}};
    }
    return {};
  };

  bool mustBufferizeInPlace(Operation* op, OpOperand& opOperand,
                            const bufferization::AnalysisState& state) const {
    return false;
  };

  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const bufferization::BufferizationOptions& options,
                          bufferization::BufferizationState& state) const {
    auto decodeOp = cast<OpTy>(op);
    FailureOr<Value> maybeBuffer =
        getBuffer(rewriter, decodeOp.getValue(), options, state);
    if (failed(maybeBuffer)) return failure();
    Value buffer = *maybeBuffer;

    bufferization::replaceOpWithNewBufferizedOp<OpTy>(
        rewriter, op, decodeOp.getEncoder(), decodeOp.getPlaintext(), buffer);

    return success();
  }
};

template <typename OpTy>
struct EncodeOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          EncodeOpInterface<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    return true;
  };

  bool bufferizesToMemoryWrite(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    return opOperand.getOperandNumber() == 2;
  };

  bufferization::AliasingValueList getAliasingValues(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    if (opOperand.getOperandNumber() == 2) {
      return {{op->getResult(0), bufferization::BufferRelation::Equivalent}};
    }
    return {};
  };

  bool mustBufferizeInPlace(Operation* op, OpOperand& opOperand,
                            const bufferization::AnalysisState& state) const {
    return false;
  };

  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const bufferization::BufferizationOptions& options,
                          bufferization::BufferizationState& state) const {
    auto encodeOp = cast<OpTy>(op);
    FailureOr<Value> maybeBuffer =
        getBuffer(rewriter, encodeOp.getValue(), options, state);
    if (failed(maybeBuffer)) return failure();
    Value buffer = *maybeBuffer;

    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(1, buffer); });

    return success();
  }
};

template <typename OpTy>
struct LinearTransformOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          LinearTransformOpInterface<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    return opOperand.getOperandNumber() == 3;
  };

  bool bufferizesToMemoryWrite(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    return false;
  };

  bufferization::AliasingValueList getAliasingValues(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    return {};
  };

  bool mustBufferizeInPlace(Operation* op, OpOperand& opOperand,
                            const bufferization::AnalysisState& state) const {
    return false;
  };

  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const bufferization::BufferizationOptions& options,
                          bufferization::BufferizationState& state) const {
    auto ltOp = cast<OpTy>(op);
    FailureOr<Value> maybeBuffer =
        getBuffer(rewriter, ltOp.getDiagonals(), options, state);
    if (failed(maybeBuffer)) return failure();
    Value buffer = *maybeBuffer;

    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(3, buffer); });

    return success();
  }
};

template <typename OpTy>
struct ScalarOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          ScalarOpInterface<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    return false;
  };

  bool bufferizesToMemoryWrite(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    return false;
  };

  bufferization::AliasingValueList getAliasingValues(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    return {};
  };

  bool mustBufferizeInPlace(Operation* op, OpOperand& opOperand,
                            const bufferization::AnalysisState& state) const {
    return false;
  };

  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const bufferization::BufferizationOptions& options,
                          bufferization::BufferizationState& state) const {
    return success();
  }
};

}  // namespace

void mlir::heir::lattigo::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, LattigoDialect* dialect) {
    CKKSDecodeOp::attachInterface<DecodeOpInterface<CKKSDecodeOp>>(*ctx);
    CKKSEncodeOp::attachInterface<EncodeOpInterface<CKKSEncodeOp>>(*ctx);

    BGVDecodeOp::attachInterface<DecodeOpInterface<BGVDecodeOp>>(*ctx);
    BGVEncodeOp::attachInterface<EncodeOpInterface<BGVEncodeOp>>(*ctx);

    CKKSRotateNewOp::attachInterface<ScalarOpInterface<CKKSRotateNewOp>>(*ctx);
    CKKSMulNewOp::attachInterface<ScalarOpInterface<CKKSMulNewOp>>(*ctx);
    CKKSAddNewOp::attachInterface<ScalarOpInterface<CKKSAddNewOp>>(*ctx);
    CKKSLinearTransformOp::attachInterface<
        LinearTransformOpInterface<CKKSLinearTransformOp>>(*ctx);
    RLWEEncryptOp::attachInterface<ScalarOpInterface<RLWEEncryptOp>>(*ctx);
  });
}
