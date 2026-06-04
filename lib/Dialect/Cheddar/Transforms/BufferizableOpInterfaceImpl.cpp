#include "lib/Dialect/Cheddar/Transforms/BufferizableOpInterfaceImpl.h"

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::heir;
using namespace mlir::heir::cheddar;

namespace {

// cheddar.encode reads a float/complex message buffer (operand #1) and produces
// an opaque plaintext (no tensor result). Bufferization just swaps the message
// operand for its buffer.
struct EncodeOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          EncodeOpInterface, EncodeOp> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    return opOperand.getOperandNumber() == 1;  // $message
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
    auto encodeOp = cast<EncodeOp>(op);
    FailureOr<Value> maybeBuffer =
        getBuffer(rewriter, encodeOp.getMessage(), options, state);
    if (failed(maybeBuffer)) return failure();

    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(1, *maybeBuffer); });
    return success();
  }
};

// cheddar.decode is destination-passing: operand #2 ($value) is the output
// buffer the decoded result aliases. Mirrors Lattigo's decode.
struct DecodeOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          DecodeOpInterface, DecodeOp> {
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
    auto decodeOp = cast<DecodeOp>(op);
    FailureOr<Value> maybeBuffer =
        getBuffer(rewriter, decodeOp.getValue(), options, state);
    if (failed(maybeBuffer)) return failure();

    bufferization::replaceOpWithNewBufferizedOp<DecodeOp>(
        rewriter, op, decodeOp.getEncoder(), decodeOp.getPlaintext(),
        *maybeBuffer);
    return success();
  }
};

// cheddar.linear_transform reads the cleartext diagonals tensor (operand #3)
// and produces an opaque ciphertext (no tensor result). Bufferization just
// swaps the diagonals operand for its buffer.
struct LinearTransformOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          LinearTransformOpInterface, LinearTransformOp> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    return opOperand.getOperandNumber() == 3;  // $diagonals
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
    auto ltOp = cast<LinearTransformOp>(op);
    FailureOr<Value> maybeBuffer =
        getBuffer(rewriter, ltOp.getDiagonals(), options, state);
    if (failed(maybeBuffer)) return failure();

    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(3, *maybeBuffer); });
    return success();
  }
};

}  // namespace

void mlir::heir::cheddar::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, CheddarDialect* dialect) {
    EncodeOp::attachInterface<EncodeOpInterface>(*ctx);
    DecodeOp::attachInterface<DecodeOpInterface>(*ctx);
    LinearTransformOp::attachInterface<LinearTransformOpInterface>(*ctx);
  });
}
