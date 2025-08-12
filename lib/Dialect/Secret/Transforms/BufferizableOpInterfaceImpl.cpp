#include "lib/Dialect/Secret/Transforms/BufferizableOpInterfaceImpl.h"

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace heir {
namespace secret {

// Bufferization of secret.separator.
struct SeparatorOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          SeparatorOpInterface, secret::SeparatorOp> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    return true;
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
    return true;
  };

  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const bufferization::BufferizationOptions& options,
                          bufferization::BufferizationState& state) const {
    auto separator = cast<secret::SeparatorOp>(op);

    SmallVector<Value> newInputs;
    for (auto input : separator.getInputs()) {
      if (isa<TensorType>(input.getType())) {
        FailureOr<Value> maybeBuffer =
            getBuffer(rewriter, input, options, state);
        if (failed(maybeBuffer)) return failure();
        newInputs.push_back(*maybeBuffer);
      }
    }

    bufferization::replaceOpWithNewBufferizedOp<secret::SeparatorOp>(
        rewriter, op, newInputs);
    return success();
  };
};

void registerBufferizableOpInterfaceExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, SecretDialect* dialect) {
    SeparatorOp::attachInterface<SeparatorOpInterface>(*ctx);
  });
}

}  // namespace secret
}  // namespace heir
}  // namespace mlir
