#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h"

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"  // from @llvm-project

#define DEBUG_TYPE "layout-propagation"

namespace mlir {
namespace heir {

using linalg::ReduceOp;
using linalg::VecmatOp;
using ::mlir::arith::AddIOp;
using ::mlir::arith::ConstantOp;
using ::mlir::arith::MulIOp;
using secret::GenericOp;
using secret::SecretType;
using secret::YieldOp;
using tensor::CollapseShapeOp;
using tensor::EmptyOp;
using tensor::ExpandShapeOp;
using tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;

#define GEN_PASS_DEF_LAYOUTPROPAGATION
#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h.inc"

struct LayoutPropagation : impl::LayoutPropagationBase<LayoutPropagation> {
  using LayoutPropagationBase::LayoutPropagationBase;

  // Top level visit method handles common logic for all ops, e.g., inserting
  // conversions.
  LogicalResult visitOperation(Operation *op);

  // Op-specific transfer functions
  LogicalResult visitOperation(AddIOp op);
  LogicalResult visitOperation(CollapseShapeOp op);
  LogicalResult visitOperation(ConstantOp op);
  LogicalResult visitOperation(EmptyOp op);
  LogicalResult visitOperation(ExpandShapeOp op);
  LogicalResult visitOperation(GenericOp op);
  LogicalResult visitOperation(MulIOp op);
  LogicalResult visitOperation(ReduceOp op);
  LogicalResult visitOperation(VecmatOp op);
  LogicalResult visitOperation(YieldOp op);
  LogicalResult visitOperation(func::FuncOp op);
  LogicalResult visitOperation(func::ReturnOp op);

  // Return true if the operand layouts are compatible for the operation, and
  // false if not. Include an InFlightDiagnostic if an operand is encountered
  // that requires a layout, but none has been set.
  std::pair<bool, std::optional<InFlightDiagnostic>>
  hasCompatibleArgumentLayouts(Operation *op);

  // Op-specific compatibility functions
  std::pair<bool, std::optional<InFlightDiagnostic>>
  hasCompatibleArgumentLayouts(ReduceOp op);
  std::pair<bool, std::optional<InFlightDiagnostic>>
  hasCompatibleArgumentLayouts(VecmatOp op);

  // Insert conversion ops to rectify incompatible operand layouts
  void rectifyIncompatibleOperandLayouts(Operation *op);

  // Return the default layout for a given type
  FailureOr<AffineMap> defaultLayoutForType(Type type);

  // Helper to pass layouts through generic ops
  void passLayoutThroughOp(Operation *op);

  // Add an op attribute denoting the layouts of the op results. Assumes the
  // assignedLayouts map contains the layout for the result SSA values already.
  void setResultLayoutAttr(Operation *op);

  void runOnOperation() override;

  DenseMap<Value, AffineMap> assignedLayouts;
  DataFlowSolver *solver;
};

void visitDebugInfo(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting: " << op->getName() << "\n");
}

void debugAssignLayout(Value value, AffineMap layout) {
  LLVM_DEBUG(llvm::dbgs() << "Assigning layout " << layout << " to value "
                          << value << "\n");
}

LogicalResult LayoutPropagation::visitOperation(Operation *op) {
  visitDebugInfo(op);

  // Note, generic ops with no operands can still produce secret results,
  // (e.g., tensor.empty which is then yielded, see below) but the secretness
  // analysis doesn't detect this as a secret result.
  //
  //  %0 = secret.generic {
  //    %2 = tensor.empty() : tensor<16x16xf32>
  //    secret.yield %2 : tensor<16x16xf32>
  //  } -> !secret.secret<tensor<16x16xf32>>
  //
  // Moreover, we handle the generic's result by processing the terminator
  // yield op, so even if the secretness analysis worked on the secret.generic,
  // we'd still need it to work on the yield, and in this case it doesn't since
  // there's no upstream dependency on a secret operand. Manage this by skipping
  // the check on YieldOp here and having a special case in the handling of
  // YieldOp that allows one to yield a value without any existing layout set.
  //
  // FIXME: rethink this! Maybe add default layouts when the layout is detected
  // to be missing? Consider type-switching on the op to set default layouts for
  // plaintext operands.
  if (!isa<func::FuncOp, func::ReturnOp, GenericOp, YieldOp>(op) &&
      !isSecret(op->getOperands(), solver) &&
      !isSecret(op->getResults(), solver)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping op " << op->getName()
                            << " with no secret operands or results\n");
    return success();
  }

  // If an operand has no layout, it may for example be produced as a plaintext
  // constant, such as a zero-valued tensor for the initializer of a reduction.
  // In this case, we assign it a default layout.
  for (auto operand : op->getOperands()) {
    if (!assignedLayouts.contains(operand)) {
      if (isa<RankedTensorType>(operand.getType())) {
        FailureOr<AffineMap> layout = defaultLayoutForType(operand.getType());
        if (failed(layout)) {
          return failure();
        }
        debugAssignLayout(operand, layout.value());
        assignedLayouts.insert({operand, layout.value()});
      }
    }
  }

  auto [compatible, diag] = hasCompatibleArgumentLayouts(op);
  if (!compatible) {
    if (diag.has_value()) {
      // An InFlightDiagnostic casts to a failure()
      return diag.value();
    }
    rectifyIncompatibleOperandLayouts(op);
  }

  return TypeSwitch<Operation *, LogicalResult>(op)
      // func ops
      .Case<func::FuncOp, func::ReturnOp>(
          [&](auto op) { return visitOperation(op); })
      // arith ops
      .Case<AddIOp, ConstantOp, MulIOp>(
          [&](auto op) { return visitOperation(op); })
      // secret ops
      .Case<GenericOp, YieldOp>([&](auto op) { return visitOperation(op); })
      // linalg ops
      .Case<VecmatOp, ReduceOp>([&](auto op) { return visitOperation(op); })
      // tensor ops
      .Case<CollapseShapeOp, ExpandShapeOp, EmptyOp>(
          [&](auto op) { return visitOperation(op); })
      .Default([&](Operation *op) { return success(); });
}

std::pair<bool, std::optional<InFlightDiagnostic>>
LayoutPropagation::hasCompatibleArgumentLayouts(Operation *op) {
  // FIXME: type switch on special case ops
  if (isa<func::FuncOp, GenericOp, YieldOp>(op)) {
    return {true, std::nullopt};
  }

  if (isa<VecmatOp>(op)) {
    // Currently only support secret vectors and plaintext matrices.
    auto vecmatOp = cast<linalg::ContractionOpInterface>(op);
    Value vec = vecmatOp.lhs();
    Value mat = vecmatOp.rhs();
    if (isSecret(mat, solver) || !isSecret(vec, solver)) {
      return {false,
              op->emitError("Only secret vectors and plaintext matrices are "
                            "supported for linalg.vecmat")};
    }

    if (!assignedLayouts.contains(vec)) {
      return {false, op->emitError("vector operand has no assigned layout")};
    }
    return {true, std::nullopt};
  }

  // By default, assume operands must all have the same layout.
  std::optional<AffineMap> firstFoundLayout;

  for (auto &operand : op->getOpOperands()) {
    if (isa<RankedTensorType>(operand.get().getType())) {
      if (!assignedLayouts.contains(operand.get())) {
        // If the operand has no layout, we can't propagate layout
        // information to the result.
        return {false, op->emitError("operand has no assigned layout")};
      }
      AffineMap layout = assignedLayouts.at(operand.get());

      if (!firstFoundLayout.has_value()) firstFoundLayout = layout;
      if (layout != firstFoundLayout.value()) {
        return {false, std::nullopt};
      }
    }
  }

  return {true, std::nullopt};
}

void LayoutPropagation::rectifyIncompatibleOperandLayouts(Operation *op) {
  mlir::IRRewriter builder(&getContext());

  LLVM_DEBUG({
    auto diag = op->emitRemark() << "Inserting layout conversion op due to "
                                    "disagreeing operand layouts";
    auto &note = diag.attachNote();
    for (auto operand : op->getOperands()) {
      std::string mapStr;
      llvm::raw_string_ostream os(mapStr);
      AffineMap operandLayout;
      if (assignedLayouts.contains(operand))
        operandLayout = assignedLayouts.at(operand);
      operandLayout.print(os);
      note << "\n- Operand: " << operand << "; Layout: " << os.str() << "\n";
    }
  });

  // Target layout is chosen arbitrarily for now. A different pass is
  // responsible for optimizing the placement and mechanics of the layout
  // conversion ops. The hasCompatibleArgumentLayouts method's failure ensures
  // a layout is present for at least two operands, and they are incompatible.
  const auto it = llvm::find_if(op->getOperands(), [this](Value pair) {
    return assignedLayouts.contains(pair);
  });
  AffineMap targetLayout = assignedLayouts.at(*it);

  for (auto &opOperand : op->getOpOperands()) {
    if (!assignedLayouts.contains(opOperand.get())) continue;
    AffineMap sourceLayout = assignedLayouts.at(opOperand.get());

    if (sourceLayout != targetLayout) {
      builder.setInsertionPoint(op);
      ConvertLayoutOp convertOp = builder.create<ConvertLayoutOp>(
          op->getLoc(), opOperand.get(), AffineMapAttr::get(sourceLayout),
          AffineMapAttr::get(targetLayout));

      // Layout of the result is the same as the target layout of the
      // conversion. Mostly this is done for consistency: all ops have an
      // attribute describing the layout of their results.
      OpBuilder builder(&getContext());
      assignedLayouts.insert({convertOp.getResult(), targetLayout});
      setResultLayoutAttr(convertOp);
      op->setOperand(opOperand.getOperandNumber(), convertOp.getResult());
    }
  }
}

LogicalResult LayoutPropagation::visitOperation(func::FuncOp op) {
  // Set a default value for each argument
  int argIndex = 0;
  for (Value arg : op.getArguments()) {
    FailureOr<AffineMap> layout = defaultLayoutForType(arg.getType());
    if (failed(layout)) {
      return failure();
    }
    debugAssignLayout(arg, layout.value());
    assignedLayouts.insert({arg, layout.value()});

    // FuncOp requires arg attributes are defined as dialect attributes,
    // so we can't use an AffineMapAttr here.
    op.setArgAttr(argIndex, tensor_ext::TensorExtDialect::kLayoutAttrName,
                  LayoutAttr::get(&getContext(), layout.value()));
    ++argIndex;
  }

  // Func result attrs are handled by the ReturnOp
  return success();
}

LogicalResult LayoutPropagation::visitOperation(func::ReturnOp op) {
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  for (OpOperand &operand : op->getOpOperands()) {
    if (!assignedLayouts.contains(operand.get())) {
      if (isSecret(operand.get(), solver)) {
        return op->emitError("secret return value has no assigned layout");
      }

      // It needs no layout.
      continue;
    }
    AffineMap layout = assignedLayouts.at(operand.get());
    func.setResultAttr(operand.getOperandNumber(),
                       tensor_ext::TensorExtDialect::kLayoutAttrName,
                       LayoutAttr::get(&getContext(), layout));
  }
  return success();
}

LogicalResult LayoutPropagation::visitOperation(GenericOp op) {
  // Every block argument has the same layout as its corresponding operand.
  for (OpOperand &operand : op->getOpOperands()) {
    if (!assignedLayouts.contains(operand.get())) {
      // Assume it is not a tensor type and doesn't need a layout.
      continue;
    }
    AffineMap layout = assignedLayouts.at(operand.get());
    BlockArgument blockArg =
        op.getRegion().getArgument(operand.getOperandNumber());
    assignedLayouts.insert({blockArg, layout});
    op.setArgAttr(operand.getOperandNumber(), "layout",
                  AffineMapAttr::get(layout));
    debugAssignLayout(blockArg, layout);
  }

  // The layout of the result of the generic op is handled when the YieldOp is
  // visited.
  return success();
}

LogicalResult LayoutPropagation::visitOperation(YieldOp op) {
  // The results of the generic op has the same layouts as the yielded values
  GenericOp generic = op->getParentOfType<GenericOp>();
  for (OpOperand &operand : op->getOpOperands()) {
    Type operandType = operand.get().getType();
    if (!assignedLayouts.contains(operand.get())) {
      // If it's a tensor type, it may be something like a tensor.empty()
      // that would not be assigned a layout earlier in the walk, because
      // it does not depend on any secret information. In this case, use the
      // default layout.
      LLVM_DEBUG(llvm::dbgs() << "No layout assigned to operand "
                              << operand.get() << ", using default layout\n");
      if (isa<RankedTensorType>(operandType)) {
        FailureOr<AffineMap> layout = defaultLayoutForType(operandType);
        if (failed(layout)) {
          return failure();
        }
        debugAssignLayout(operand.get(), layout.value());
        assignedLayouts.insert({operand.get(), layout.value()});
      } else {
        // Assume it is not a tensor type and doesn't need a layout.
        continue;
      }
    }
    AffineMap layout = assignedLayouts.at(operand.get());
    Value result = generic.getResult(operand.getOperandNumber());
    assignedLayouts.insert({result, layout});
    debugAssignLayout(result, layout);
  }
  setResultLayoutAttr(generic);
  return success();
}

void LayoutPropagation::passLayoutThroughOp(Operation *op) {
  // All inputs have the same layout, so just propagate it to all results
  for (Value result : op->getResults()) {
    if (isa<RankedTensorType>(result.getType())) {
      AffineMap layout = assignedLayouts.at(op->getOperand(0));
      assignedLayouts.insert({result, layout});
      debugAssignLayout(result, layout);
    }
  }
  setResultLayoutAttr(op);
}

LogicalResult LayoutPropagation::visitOperation(ConstantOp op) {
  // Constant ops can take any layout, but to start they are implemented to have
  // row-major layouts. But if a later pass back-propagates a layout from a
  // later op, an EmptyOp can trivially take on that changed layout.
  Value result = op.getResult();
  FailureOr<AffineMap> layout = defaultLayoutForType(result.getType());
  if (failed(layout)) {
    return failure();
  }
  debugAssignLayout(result, layout.value());
  assignedLayouts.insert({result, layout.value()});
  return success();
}

LogicalResult LayoutPropagation::visitOperation(EmptyOp op) {
  // Empty ops can take any layout, but to start they are implemented to have
  // row-major layouts. But if a later pass back-propagates a layout from a
  // later op, an EmptyOp can trivially take on that changed layout.
  Value result = op.getResult();
  FailureOr<AffineMap> layout = defaultLayoutForType(result.getType());
  if (failed(layout)) {
    return failure();
  }
  debugAssignLayout(result, layout.value());
  assignedLayouts.insert({result, layout.value()});
  return success();
}

LogicalResult LayoutPropagation::visitOperation(CollapseShapeOp op) {
  // Only support rank-reduced types for now, i.e., where the collapsed
  // shape only removes static dimensions of size 1.
  SliceVerificationResult res =
      isRankReducedType(op.getSrcType(), op.getResultType());
  if (res != SliceVerificationResult::Success)
    return op->emitError(
        "Only rank-reduced types are supported for CollapseShapeOp");

  auto tensor = op.getSrc();
  AffineMap inputLayout = assignedLayouts.at(tensor);
  unsigned numDims = tensor.getType().getRank();
  llvm::SmallBitVector dimsBV(numDims, false);

  for (Attribute associationGroup : op.getReassociation()) {
    auto associationArray = dyn_cast<ArrayAttr>(associationGroup).getValue();
    // a single-entry association group is a no-op
    if (associationArray.size() == 1) {
      continue;
    }
    for (Attribute association : associationArray) {
      int64_t reassocDim = cast<IntegerAttr>(association).getInt();
      if (op.getSrcType().getShape()[reassocDim] == 1) dimsBV.set(reassocDim);
    }
  }

  AffineMap resultLayout =
      projectDims(inputLayout, dimsBV, /*compressDims=*/true);
  assignedLayouts.insert({op.getResult(), resultLayout});
  setResultLayoutAttr(op);
  debugAssignLayout(op.getResult(), resultLayout);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(ExpandShapeOp op) {
  MLIRContext *context = &getContext();
  // Only support rank-reduced types for now, i.e., where the expanded shape
  // only adds static dimensions of size 1.
  SliceVerificationResult res =
      isRankReducedType(op.getResultType(), op.getSrcType());
  if (res != SliceVerificationResult::Success)
    return op->emitError(
        "Only rank-reduced types are supported for ExpandShapeOp");

  auto tensor = op.getSrc();
  AffineMap inputLayout = assignedLayouts.at(tensor);

  // tensor indices correspond to layout dimensions, and adding a dimension of
  // size 1 has no effect on the affine map expressions, so all we're doing is
  // adding new dimensions for each reassociation group index corresponding to
  // an output dimension of size 1. Mainly we have to ensure that the dimension
  // we're adding is in the correct index of the affine map's dimension list.
  int oldDim = 0;
  DenseMap<AffineExpr, AffineExpr> oldDimsToNewDims;
  for (Attribute associationGroup : op.getReassociation()) {
    auto associationArray = dyn_cast<ArrayAttr>(associationGroup).getValue();
    // a single-entry association group is a no-op
    if (associationArray.size() == 1) {
      oldDimsToNewDims[getAffineDimExpr(oldDim, context)] = getAffineDimExpr(
          cast<IntegerAttr>(associationArray[0]).getInt(), context);
      ++oldDim;
      continue;
    }

    for (Attribute association : associationArray) {
      int64_t reassocDim = cast<IntegerAttr>(association).getInt();
      if (op.getResultType().getShape()[reassocDim] > 1) {
        oldDimsToNewDims[getAffineDimExpr(oldDim, context)] =
            getAffineDimExpr(reassocDim, context);
        ++oldDim;
      }
    }
  }

  int resultNumDims = op.getResultType().getRank();
  // First create a larger-rank affine map, but using old dimension identifiers
  AffineMap resLayout1 =
      AffineMap::get(resultNumDims, /*symbolCount=*/0, inputLayout.getResults(),
                     &getContext());

  // Then replace the old dimension identifier expressions with new ones
  AffineMap resultLayout = resLayout1.replace(oldDimsToNewDims);

  assignedLayouts.insert({op.getResult(), resultLayout});
  setResultLayoutAttr(op);
  debugAssignLayout(op.getResult(), resultLayout);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(VecmatOp op) {
  auto vecmatOp = cast<linalg::ContractionOpInterface>(*op);
  auto vec = vecmatOp.lhs();

  // The matrix has no assigned layout because it is assumed to be
  // plaintext/static (this is intended to be enforced by
  // hasCompatibleArgumentLayouts).
  AffineMap vecLayout = assignedLayouts.at(vec);

  // Always one result, and it's a vector with the same layout
  // as the input vector
  auto result = vecmatOp->getResult(0);
  AffineMap resultLayout = vecLayout;

  assignedLayouts.insert({result, resultLayout});
  setResultLayoutAttr(op);
  debugAssignLayout(result, resultLayout);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(AddIOp op) {
  passLayoutThroughOp(op);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(MulIOp op) {
  passLayoutThroughOp(op);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(ReduceOp op) {
  // Reduce has a nested region, but for now we only support the special
  // cases that correspond to the "shortened print form" listed at
  // https://mlir.llvm.org/docs/Dialects/Linalg/#linalgreduce-linalgreduceop
  //
  // I.e., the body of the reduce op is a single scalar operation that
  // takes as its first input the initializer value of the reduction.
  ArrayRef<int64_t> dimsToSum = op.getDimensions();

  if (op.getInputs().size() != 1) {
    return op->emitError("Only support reductions with a single input");
  }
  Value tensor = op.getInputs()[0];
  Value result = op.getResult(0);

  AffineMap inputLayout = assignedLayouts.at(tensor);
  // The result layout is equivalent to reducing the summed dimensions
  // to 1 and then dropping them.

  unsigned numDims = cast<ShapedType>(tensor.getType()).getRank();
  llvm::SmallBitVector dimsBV(numDims, false);
  for (int dimToSum : dimsToSum) dimsBV.set(dimToSum);

  AffineMap resultLayout =
      projectDims(inputLayout, dimsBV, /*compressDims=*/true);

  assignedLayouts.insert({result, resultLayout});
  setResultLayoutAttr(op);
  debugAssignLayout(result, resultLayout);
  return success();
}

FailureOr<AffineMap> LayoutPropagation::defaultLayoutForType(Type type) {
  Type ty = type;
  if (SecretType secretType = dyn_cast<SecretType>(type)) {
    ty = secretType.getValueType();
  }

  // RankedTensorType is laid out by default in row-major order
  if (RankedTensorType tensorType = dyn_cast<RankedTensorType>(ty)) {
    unsigned rank = tensorType.getRank();
    ArrayRef<int64_t> shape = tensorType.getShape();
    SmallVector<AffineExpr, 4> dims;
    for (unsigned i = 0; i < rank; ++i) {
      dims.push_back(getAffineDimExpr(i, type.getContext()));
    }

    // For a tensor of type tensor<n1xn2xi16>, the row-major layout
    // would be represented by the AffineMap:
    //
    //  (d0, d1) -> (d0 * n2 + d1)
    //
    // For a 3-dimension tensor of shape (n1, n2, n3), it would be
    //
    //  (d0, d1, d2) -> (d0 * n2 * n3 + d1 * n3 + d2)
    //
    // And so on.
    AffineExpr expr = dims[0];
    for (unsigned i = 1; i < rank; ++i) {
      expr = expr * shape[i] + dims[i];
    }

    return AffineMap::get(rank, /*symbolCount=*/0, expr);
  }

  return failure();
}

void LayoutPropagation::setResultLayoutAttr(Operation *op) {
  OpBuilder builder(&getContext());
  SmallVector<AffineMap> resultLayouts = llvm::map_to_vector(
      op->getResults(),
      [&](Value result) { return assignedLayouts.at(result); });
  op->setAttr("layout", builder.getAffineMapArrayAttr(resultLayouts));
}

void LayoutPropagation::runOnOperation() {
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<SecretnessAnalysis>();
  if (failed(solver.initializeAndRun(getOperation()))) {
    getOperation()->emitOpError() << "Failed to run secretness analysis.\n";
    signalPassFailure();
    return;
  }
  this->solver = &solver;

  LLVM_DEBUG(llvm::dbgs() << "Running layout propagation on operation: "
                          << getOperation()->getName() << "\n");
  WalkResult result =
      getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
        LogicalResult result = visitOperation(op);
        if (failed(result)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

  if (result.wasInterrupted()) {
    signalPassFailure();
  }
};

}  // namespace heir
}  // namespace mlir
