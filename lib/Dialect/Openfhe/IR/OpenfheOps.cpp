#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

// ElementwiseByOperandOpInterface methods

bool FastRotationOp::operandIsMappable(unsigned operandIndex) {
  // All operands are wholesale, just the index attributes are mappable.
  return false;
}

// BatchVectorizableOpInterface methods

bool FastRotationOp::isBatchCompatible(Operation* rhs) {
  FastRotationOp rhsRotate = dyn_cast<FastRotationOp>(rhs);
  if (!rhsRotate) return false;
  // Only different rotations of the same ciphertext can be batched.
  return (getCryptoContext() == rhsRotate.getCryptoContext() &&
          getInput() == rhsRotate.getInput() &&
          getPrecomputedDigitDecomp() ==
              rhsRotate.getPrecomputedDigitDecomp() &&
          getCyclotomicOrder() == rhsRotate.getCyclotomicOrder());
}

FailureOr<Operation*> FastRotationOp::buildBatchedOperation(
    MLIRContext* context, OpBuilder& builder,
    SmallVector<Value> vectorizedOperands,
    SmallVector<Operation*> batchedOperations) {
  SmallVector<Value> indexVals;
  for (auto* op : batchedOperations) {
    if (auto rotateOp = dyn_cast<FastRotationOp>(op)) {
      indexVals.push_back(rotateOp.getIndex());
    } else {
      op->emitOpError("unsupported operation for vectorization");
      return failure();
    }
  }
  auto indexTensor = tensor::FromElementsOp::create(
      builder, getLoc(),
      RankedTensorType::get({static_cast<int64_t>(batchedOperations.size())},
                            builder.getIndexType()),
      indexVals);

  Location loc = getLoc();
  int64_t batchSize = batchedOperations.size();
  Type ctType = getResult().getType();
  auto initTensor = tensor::EmptyOp::create(
      builder, loc, ArrayRef<int64_t>({batchSize}), ctType);

  std::vector<OpFoldResult> lowerBounds = {builder.getIndexAttr(0)};
  std::vector<OpFoldResult> upperBounds = {builder.getIndexAttr(batchSize)};
  std::vector<OpFoldResult> steps = {builder.getIndexAttr(1)};
  Operation* forallOp = scf::ForallOp::create(
      builder, loc, lowerBounds, upperBounds, steps,
      /*outputs=*/ValueRange{initTensor},
      /*mapping=*/std::nullopt,
      /*bodyBuilder=*/
      [&](OpBuilder& b, Location loc, ValueRange regionArgs) {
        Value j = regionArgs[0];
        Value sharedOut = regionArgs.take_back()[0];
        Value idx =
            tensor::ExtractOp::create(b, loc, indexTensor, ValueRange{j});

        auto rotateOp = openfhe::FastRotationOp::create(
            b, loc, ctType, getCryptoContext(), getInput(), idx,
            getCyclotomicOrder(), getPrecomputedDigitDecomp());
        Value rotatedVal = rotateOp.getResult();
        Value slice = tensor::FromElementsOp::create(b, loc, rotatedVal);

        auto term = scf::InParallelOp::create(b, loc);
        b.setInsertionPointToStart(term.getBody());
        tensor::ParallelInsertSliceOp::create(
            b, loc, slice, sharedOut,
            /*offsets=*/SmallVector<OpFoldResult>{j},
            /*sizes=*/SmallVector<OpFoldResult>{b.getIndexAttr(1)},
            /*strides=*/SmallVector<OpFoldResult>{b.getIndexAttr(1)});
      });
  return forallOp;
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
