#ifndef LIB_DIALECT_UTILS_H_
#define LIB_DIALECT_UTILS_H_

#include <cstdint>

#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"            // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

namespace mlir {
namespace heir {

/// Given a tensor::InsertOp or tensor::ExtractOp, and assuming the shape
/// of the input tensor is 1-dimensional and the input index is constant,
/// return the constant index value. If any of these conditions are not
/// met, return a failure.
template <typename Op>
FailureOr<int64_t> get1DExtractionIndex(Op op) {
  auto insertIndices = op.getIndices();
  if (insertIndices.size() != 1) return failure();

  // Each index must be constant; this may require running --canonicalize or
  // -sccp before this pass to apply folding rules (use -sccp if you need to
  // fold constants through control flow).
  Value insertIndex = *insertIndices.begin();
  auto insertIndexConstOp =
      insertIndex.getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (!insertIndexConstOp) return failure();

  auto insertOffsetAttr =
      llvm::dyn_cast<IntegerAttr>(insertIndexConstOp.getValue());
  if (!insertOffsetAttr) return failure();

  return insertOffsetAttr.getInt();
}

inline Operation* cloneWithNewResultTypes(Operation* op,
                                          TypeRange newResultTypes,
                                          IRMapping& mapper) {
  SmallVector<Value, 8> operands;
  SmallVector<Block*, 2> successors;

  // Remap the operands.
  operands.reserve(op->getNumOperands());
  for (auto opValue : op->getOperands())
    operands.push_back(mapper.lookupOrDefault(opValue));

  // Remap the successors.
  successors.reserve(op->getNumSuccessors());
  for (Block* successor : op->getSuccessors())
    successors.push_back(mapper.lookupOrDefault(successor));

  // Create the new operation.
  auto* newOp = Operation::create(
      op->getLoc(), op->getName(), newResultTypes, operands, op->getAttrs(),
      op->getPropertiesStorage(), successors, op->getNumRegions());
  mapper.map(op, newOp);

  // Clone the regions.
  for (unsigned i = 0; i != op->getNumRegions(); ++i)
    op->getRegion(i).cloneInto(&newOp->getRegion(i), mapper);

  // Remember the mapping of any results.
  for (unsigned i = 0, e = op->getNumResults(); i != e; ++i)
    mapper.map(op->getResult(i), newOp->getResult(i));

  return newOp;
}

inline Operation* cloneWithNewResultTypes(Operation* op,
                                          TypeRange newResultTypes) {
  IRMapping mapper;
  return cloneWithNewResultTypes(op, newResultTypes, mapper);
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_UTILS_H_
