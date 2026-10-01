#include "lib/Dialect/Preprocessing/Transforms/BufferizableOpInterfaceImpl.h"

#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

struct LoadResourceOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          LoadResourceOpInterface, LoadResourceOp> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    auto loadResourceOp = cast<LoadResourceOp>(op);
    return loadResourceOp.isDpsInit(&opOperand);
  }

  bufferization::AliasingValueList getAliasingValues(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    auto loadResourceOp = cast<LoadResourceOp>(op);
    if (loadResourceOp.isDpsInit(&opOperand)) {
      return {{loadResourceOp.getTiedOpResult(&opOperand),
               bufferization::BufferRelation::Equivalent}};
    }
    return {};
  }

  bool isWritable(Operation* op, Value value,
                  const bufferization::AnalysisState& state) const {
    return false;
  }

  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const bufferization::BufferizationOptions& options,
                          bufferization::BufferizationState& state) const {
    auto loadResourceOp = cast<LoadResourceOp>(op);
    FailureOr<Value> destination = bufferization::getBuffer(
        rewriter, loadResourceOp.getDestination(), options, state);
    if (failed(destination)) return failure();

    LoadResourceOp::create(rewriter, op->getLoc(), TypeRange{},
                           loadResourceOp.getPathAttr(), *destination);
    bufferization::replaceOpWithBufferizedValues(rewriter, op, *destination);
    return success();
  }
};

void registerBufferizableOpInterfaceExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, PreprocessingDialect* dialect) {
    LoadResourceOp::attachInterface<LoadResourceOpInterface>(*ctx);
  });
}

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir
