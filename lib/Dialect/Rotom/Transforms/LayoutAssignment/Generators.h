#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_GENERATORS_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_GENERATORS_H_

#include <optional>

#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/AssignmentContext.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir::heir::rotom {

// One candidate-layout generator per tensor op the layout assignment supports.
// Each reads/writes candidates and queries costs only through the
// AssignmentContext, so it is independent of the pass that drives it.
// LayoutAssignment::visitOperation dispatches to these by op type.

// Structural (gen/Structural.cpp): values that carry layouts unchanged.
LogicalResult generateFunc(AssignmentContext& ctx, func::FuncOp op);
LogicalResult generateSecretGeneric(AssignmentContext& ctx,
                                    secret::GenericOp op);
LogicalResult generateYield(AssignmentContext& ctx, secret::YieldOp op);
// Bring all operands to a common layout; the dispatch default and the fallback
// the shape generators reach for when an op's dim map is unsupported.
LogicalResult generatePassThrough(AssignmentContext& ctx, Operation* op);

// Elementwise (gen/Binop.cpp): convert-then-compute binary ops.
LogicalResult generateElementwise(AssignmentContext& ctx, Operation* op);
LogicalResult generateLinalgGeneric(AssignmentContext& ctx,
                                    linalg::GenericOp op);

// Reduce / transpose (gen/ReduceTranspose.cpp): dim-permuting and dim-dropping.
LogicalResult generateTranspose(AssignmentContext& ctx, linalg::TransposeOp op);
LogicalResult generateReduction(AssignmentContext& ctx, linalg::ReduceOp op);

// Matmul (gen/Binop.cpp): the reference's binop pipeline for
// matmul -- replicate each operand to the aligned shape, apply the sum roll
// on either side, align the pair (a conversion or a roll), derive the result.
// Every step is priced by estimateConversionCost, which covers conversions
// and rolls alike; the lowering replays the same steps as rotom.convert_layout
// and rotom.apply_roll ops, so no kernel name is attached.
LogicalResult generateMatmul(AssignmentContext& ctx, linalg::MatmulOp op);
// The cheapest alignment generateMatmul priced for one (lhs, rhs, result)
// layout combination, re-derived deterministically for the rewrite. Each
// chain runs from the operand's assigned layout to its aligned placement,
// one kernel per step (a conversion, or a roll when the step adds one).
struct MatmulAlignment {
  SmallVector<LayoutAttr> lhsChain;
  SmallVector<LayoutAttr> rhsChain;
  LayoutAttr compute;
  LayoutAttr result;
};
std::optional<MatmulAlignment> selectMatmulAlignment(
    AssignmentContext& ctx, LayoutAttr lhs, LayoutAttr rhs, LayoutAttr result,
    ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape);

// Reshape / slice (gen/Reshape.cpp): dim collapse/expand and slice
// insert/extract.
LogicalResult generateCollapseShape(AssignmentContext& ctx,
                                    tensor::CollapseShapeOp op);
LogicalResult generateExpandShape(AssignmentContext& ctx,
                                  tensor::ExpandShapeOp op);
LogicalResult generateExtractSlice(AssignmentContext& ctx,
                                   tensor::ExtractSliceOp op);
LogicalResult generateInsertSlice(AssignmentContext& ctx,
                                  tensor::InsertSliceOp op);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_GENERATORS_H_
