#include "lib/Dialect/Preprocessing/Transforms/BufferizableOpInterfaceImpl.h"

#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

// load_resource requires special bufferization because in the backend it
// corresponds to populating static global data with bytes from a file, and
// bufferizing this requires us to avoid inserting deallocations.
struct LoadResourceOpInterface
    : public ::mlir::bufferization::BufferizableOpInterface::ExternalModel<
          LoadResourceOpInterface, LoadResourceOp> {
  LogicalResult bufferize(
      Operation* op, RewriterBase& rewriter,
      const ::mlir::bufferization::BufferizationOptions& options,
      ::mlir::bufferization::BufferizationState& state) const {
    auto loadResourceOp = cast<LoadResourceOp>(op);
    auto tensorType = dyn_cast<RankedTensorType>(loadResourceOp.getType());

    if (!tensorType) return success();

    Attribute memorySpace;
    if (auto memSpace = options.defaultMemorySpaceFn(
            cast<::mlir::bufferization::TensorLikeType>(tensorType)))
      memorySpace = *memSpace;
    else
      return op->emitError("could not infer memory space");

    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                        MemRefLayoutAttrInterface(), memorySpace);

    ::mlir::bufferization::replaceOpWithNewBufferizedOp<LoadResourceOp>(
        rewriter, op, memrefType, loadResourceOp.getPathAttr());
    return success();
  }

  // Everything else is boilerplate for this core step: telling MLIR it cannot
  // write to this buffer.
  bool isWritable(Operation* op, Value value,
                  const ::mlir::bufferization::AnalysisState& state) const {
    return false;
  }

  FailureOr<::mlir::bufferization::BufferLikeType> getBufferType(
      Operation* op, Value value,
      const ::mlir::bufferization::BufferizationOptions& options,
      const ::mlir::bufferization::BufferizationState& state,
      SmallVector<Value>& invocationStack) const {
    auto loadResourceOp = cast<LoadResourceOp>(op);
    auto tensorType = dyn_cast<RankedTensorType>(loadResourceOp.getType());
    if (!tensorType) return failure();

    Attribute memorySpace;
    if (auto memSpace = options.defaultMemorySpaceFn(
            cast<::mlir::bufferization::TensorLikeType>(tensorType)))
      memorySpace = *memSpace;
    else
      return op->emitError("could not infer memory space");

    return cast<::mlir::bufferization::BufferLikeType>(
        MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                        MemRefLayoutAttrInterface(), memorySpace));
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
