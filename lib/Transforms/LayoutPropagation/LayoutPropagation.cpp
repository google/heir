#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h"

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"    // from @llvm-project

#define DEBUG_TYPE "layout-propagation"

namespace mlir {
namespace heir {

using ::mlir::arith::AddIOp;
using ::mlir::arith::MulIOp;
using secret::GenericOp;
using secret::SecretType;
using secret::YieldOp;
using tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;
using tensor_ext::SumOp;

#define GEN_PASS_DEF_LAYOUTPROPAGATION
#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h.inc"

struct LayoutPropagation : impl::LayoutPropagationBase<LayoutPropagation> {
  using LayoutPropagationBase::LayoutPropagationBase;

  // Top level visit method handles common logic for all ops, e.g., inserting
  // conversions.
  LogicalResult visitOperation(Operation *op);

  // Op-specific transfer functions
  LogicalResult visitOperation(func::FuncOp op);
  LogicalResult visitOperation(AddIOp op);
  LogicalResult visitOperation(MulIOp op);
  LogicalResult visitOperation(GenericOp op);
  LogicalResult visitOperation(YieldOp op);
  LogicalResult visitOperation(SumOp op);

  // Return the default layout for a given type
  FailureOr<AffineMap> defaultLayoutForType(Type type);

  // Helper to pass layouts through generic ops
  void passLayoutThroughOp(Operation *op);

  // Add an op attribute denoting the layouts of the op results. Assumes the
  // assignedLayouts map contains the layout for the result SSA values already.
  void setResultLayoutAttr(Operation *op);

  void runOnOperation() override;

  DenseMap<Value, AffineMap> assignedLayouts;
};

void visitDebugInfo(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting: " << op->getName() << "\n");
}

void debugAssignLayout(Value value, AffineMap layout) {
  LLVM_DEBUG(llvm::dbgs() << "Assigning layout " << layout << " to value "
                          << value << "\n");
}

// A helper struct to keep track of an operand's index and layout.
struct OperandLayout {
  Value operand;
  unsigned int index;
  AffineMap layout;
};

LogicalResult LayoutPropagation::visitOperation(Operation *op) {
  visitDebugInfo(op);
  mlir::IRRewriter builder(&getContext());
  // When the operands have different layout attributes, it is invalid
  // to apply the op, and we must insert layout conversion ops.
  SmallVector<OperandLayout> layouts;
  bool disagreeingLayouts = false;

  // FIXME: file an issue to smartly choose the targetLayout, e.g., using
  // the most common shared layout, or analyze for the cheapest conversion.
  std::optional<AffineMap> targetLayout;

  for (auto &operand : op->getOpOperands()) {
    if (isa<RankedTensorType>(operand.get().getType())) {
      if (assignedLayouts.count(operand.get()) == 0) {
        // If the operand has no layout, we can't propagate layout
        // information to the result.
        return op->emitError("operand has no assigned layout");
      }
      AffineMap layout = assignedLayouts[operand.get()];
      OperandLayout operandLayout = {operand.get(), operand.getOperandNumber(),
                                     layout};
      layouts.push_back(operandLayout);

      if (!targetLayout.has_value()) targetLayout = layout;

      if (layout != targetLayout.value()) {
        disagreeingLayouts = true;
      }
    }
  }

  // If the operands have different layouts, we need to insert layout
  // conversion ops.
  if (!layouts.empty() && disagreeingLayouts) {
    LLVM_DEBUG({
      auto diag = op->emitRemark() << "Inserting layout conversion op due to "
                                      "disagreeing operand layouts";
      auto &note = diag.attachNote();
      for (auto operandLayout : layouts) {
        std::string mapStr;
        llvm::raw_string_ostream os(mapStr);
        operandLayout.layout.print(os);
        note << "\n- Operand: " << operandLayout.operand
             << "; Layout: " << os.str() << "\n";
      }
    });

    for (auto &operandLayout : layouts) {
      Value operand = operandLayout.operand;
      AffineMap sourceLayout = operandLayout.layout;
      if (sourceLayout != targetLayout) {
        builder.setInsertionPoint(op);
        ConvertLayoutOp convertOp = builder.create<ConvertLayoutOp>(
            op->getLoc(), operand, AffineMapAttr::get(sourceLayout),
            AffineMapAttr::get(targetLayout.value()));

        // Layout of the result is the same as the target layout of the
        // conversion. Mostly this is done for consistency: all ops have an
        // attribute describing the layout of their results.
        convertOp->setAttr("layout", AffineMapAttr::get(targetLayout.value()));
        op->setOperand(operandLayout.index, convertOp.getResult());
      }
    }
    op->getParentOp()->dump();
  }

  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<func::FuncOp>([&](auto op) { return visitOperation(op); })
      .Case<AddIOp, MulIOp>([&](auto op) { return visitOperation(op); })
      .Case<GenericOp, YieldOp>([&](auto op) { return visitOperation(op); })
      .Case<SumOp>([&](auto op) { return visitOperation(op); })
      .Default([&](Operation *op) { return success(); });
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
    assignedLayouts.insert(std::make_pair(arg, layout.value()));

    // FuncOp requires arg attributes are defined as dialect attributes,
    // so we can't use an AffineMapAttr here.
    op.setArgAttr(argIndex, tensor_ext::TensorExtDialect::kLayoutAttrName,
                  LayoutAttr::get(&getContext(), layout.value()));
    ++argIndex;
  }

  return success();
}

LogicalResult LayoutPropagation::visitOperation(GenericOp op) {
  // Every block argument has the same layout as its corresponding operand.
  for (OpOperand &operand : op->getOpOperands()) {
    if (assignedLayouts.count(operand.get()) == 0) {
      // Assume it is not a tensor type and doesn't need a layout.
      continue;
    }
    AffineMap layout = assignedLayouts[operand.get()];
    BlockArgument blockArg =
        op.getRegion().getArgument(operand.getOperandNumber());
    assignedLayouts.insert(std::make_pair(blockArg, layout));
    op.setArgAttr(operand.getOperandNumber(), "layout",
                  AffineMapAttr::get(layout));
    debugAssignLayout(operand.get(), layout);
  }

  // The layout of the result of the generic op is handled when the YieldOp is
  // visited.
  return success();
}

LogicalResult LayoutPropagation::visitOperation(YieldOp op) {
  // The results of the generic op has the same layouts as the yielded values
  GenericOp generic = op->getParentOfType<GenericOp>();
  for (OpOperand &operand : op->getOpOperands()) {
    if (assignedLayouts.count(operand.get()) == 0) {
      // Assume it is not a tensor type and doesn't need a layout.
      continue;
    }
    AffineMap layout = assignedLayouts[operand.get()];
    Value result = generic.getResult(operand.getOperandNumber());
    assignedLayouts.insert(std::make_pair(result, layout));
    debugAssignLayout(result, layout);
  }
  setResultLayoutAttr(generic);
  return success();
}

void LayoutPropagation::passLayoutThroughOp(Operation *op) {
  // All inputs have the same layout, so just propagate it to all results
  for (Value result : op->getResults()) {
    if (isa<RankedTensorType>(result.getType())) {
      AffineMap layout = assignedLayouts[op->getOperand(0)];
      assignedLayouts.insert(std::make_pair(result, layout));
      debugAssignLayout(result, layout);
    }
  }
  setResultLayoutAttr(op);
}

LogicalResult LayoutPropagation::visitOperation(arith::AddIOp op) {
  passLayoutThroughOp(op);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(arith::MulIOp op) {
  passLayoutThroughOp(op);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(tensor_ext::SumOp op) {
  unsigned dimToSum = op.getDim().getZExtValue();
  Value tensor = op.getTensor();
  Value result = op.getOutput();

  AffineMap inputLayout = assignedLayouts[tensor];
  // The result layout is equivalent to reducing the summed dimension
  // to 1 and then dropping it.

  unsigned numDims = cast<ShapedType>(tensor.getType()).getRank();
  llvm::SmallBitVector dimsBV(numDims, false);
  dimsBV.set(dimToSum);

  AffineMap resultLayout =
      projectDims(inputLayout, dimsBV, /*compressDims=*/true);

  assignedLayouts.insert(std::make_pair(result, resultLayout));
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
      op->getResults(), [&](Value result) { return assignedLayouts[result]; });
  op->setAttr("layout", builder.getAffineMapArrayAttr(resultLayouts));
}

void LayoutPropagation::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);

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
