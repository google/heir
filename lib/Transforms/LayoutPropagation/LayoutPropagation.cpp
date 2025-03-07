#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h"

#include <cstdint>
#include <optional>
#include <string>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"             // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

#define DEBUG_TYPE "layout-propagation"

namespace mlir {
namespace heir {

using linalg::ReduceOp;
using linalg::VecmatOp;
using ::mlir::arith::AddIOp;
using ::mlir::arith::MulIOp;
using secret::GenericOp;
using secret::SecretType;
using secret::YieldOp;
using tensor::CollapseShapeOp;
using tensor::ExpandShapeOp;
using tensor_ext::AssignLayoutOp;
using tensor_ext::ConvertLayoutOp;

#define GEN_PASS_DEF_LAYOUTPROPAGATION
#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h.inc"

// The result of a compatibility check for the layouts of an op's operands (cf.
// hasCompatibleArgumentLayouts). If the check fails, the presence of a
// diagnostic signals that the failure is unrecoverable and should cause the
// pass to fail. If the diagnostic is nullopt, then the failure can be
// recovered by rectifyIncompatibleOperandLayouts.
struct CompatibilityResult {
  bool compatible;
  std::optional<InFlightDiagnostic> diag;
};

struct LayoutPropagation : impl::LayoutPropagationBase<LayoutPropagation> {
  using LayoutPropagationBase::LayoutPropagationBase;

  // Top level visit method handling common logic and dispatching to specific
  // visitOperation overloads.
  LogicalResult visitOperation(Operation *op);

  // Op-specific transfer functions
  LogicalResult visitOperation(AddIOp op);
  LogicalResult visitOperation(CollapseShapeOp op);
  LogicalResult visitOperation(ExpandShapeOp op);
  LogicalResult visitOperation(GenericOp op);
  LogicalResult visitOperation(MulIOp op);
  LogicalResult visitOperation(ReduceOp op);
  LogicalResult visitOperation(VecmatOp op);
  LogicalResult visitOperation(YieldOp op);
  LogicalResult visitOperation(func::FuncOp op);
  LogicalResult visitOperation(func::ReturnOp op);

  // Determine if the operation arguments have compatible layouts for the given
  // op. If the check fails, the CompatibilityResult::compatible field is
  // false. If there is also a diagnostic populated in the result, the failure
  // is unrecoverable.
  CompatibilityResult hasCompatibleArgumentLayouts(Operation *op);

  // Op-specific compatibility functions
  CompatibilityResult hasCompatibleArgumentLayouts(ReduceOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(VecmatOp op);

  // Insert conversion ops to rectify incompatible operand layouts
  void rectifyIncompatibleOperandLayouts(Operation *op);

  // Op-specific overrides
  void rectifyIncompatibleOperandLayouts(ReduceOp op);

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

// A helper to convert the layout of an input tensor to a reduce op. The result
// layout is equivalent to reducing the summed dimensions to 1 and then
// dropping them.
//
// TODO(1352): Determine if/how to support repetition in the layout.
AffineMap convertLayoutForReduce(AffineMap inputLayout,
                                 ArrayRef<int64_t> dimsToReduce) {
  unsigned numDims = inputLayout.getNumDims();
  llvm::SmallBitVector dimsBV(numDims, false);
  for (int dimToSum : dimsToReduce) dimsBV.set(dimToSum);
  return projectDims(inputLayout, dimsBV, /*compressDims=*/true);
}

LogicalResult LayoutPropagation::visitOperation(Operation *op) {
  visitDebugInfo(op);

  if (!isa<func::FuncOp, func::ReturnOp, GenericOp, YieldOp>(op) &&
      !isSecret(op->getOperands(), solver) &&
      !isSecret(op->getResults(), solver)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping op " << op->getName()
               << " because its operands and results are non-secret, or it is "
                  "in a special allowlist of ops to ignore\n");
    return success();
  }

  // If an operand has no layout, it may for example be produced as a plaintext
  // constant, such as a zero-valued tensor for the initializer of a reduction.
  // In this case, we insert a layout assignment.
  for (auto operand : op->getOperands()) {
    if (!assignedLayouts.contains(operand)) {
      if (isa<RankedTensorType>(operand.getType())) {
        LLVM_DEBUG(llvm::dbgs() << "tensor operand " << operand
                                << " has no layout assigned\n");
        FailureOr<AffineMap> layout = defaultLayoutForType(operand.getType());
        if (failed(layout)) {
          return failure();
        }
        mlir::IRRewriter builder(&getContext());
        builder.setInsertionPoint(op);
        AssignLayoutOp assignLayoutOp = builder.create<AssignLayoutOp>(
            op->getLoc(), operand, AffineMapAttr::get(layout.value()));
        Value toReplace = assignLayoutOp.getResult();
        // This may create duplicate layout assignment ops, and we expect CSE
        // to later clean them up. Otherwise we risk replacing a use of the
        // cleartext value in some other context.
        builder.replaceUsesWithIf(operand, toReplace, [&](OpOperand &operand) {
          return operand.getOwner() == op;
        });
        debugAssignLayout(toReplace, layout.value());
        assignedLayouts.insert({toReplace, layout.value()});
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
      .Case<AddIOp, MulIOp>([&](auto op) { return visitOperation(op); })
      // secret ops
      .Case<GenericOp, YieldOp>([&](auto op) { return visitOperation(op); })
      // linalg ops
      .Case<VecmatOp, ReduceOp>([&](auto op) { return visitOperation(op); })
      // tensor ops
      .Case<CollapseShapeOp, ExpandShapeOp>(
          [&](auto op) { return visitOperation(op); })
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
    assignedLayouts.insert({arg, layout.value()});
    op.setArgAttr(argIndex, tensor_ext::TensorExtDialect::kLayoutAttrName,
                  AffineMapAttr::get(layout.value()));
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
                       AffineMapAttr::get(layout));
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
    op.setOperandAttr(operand.getOperandNumber(), "layout",
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
  for (const auto &[tensor, result] :
       llvm::zip(op.getInputs(), op.getResults())) {
    AffineMap resultLayout =
        convertLayoutForReduce(assignedLayouts.at(tensor), op.getDimensions());
    assignedLayouts.insert({result, resultLayout});
    debugAssignLayout(result, resultLayout);
  }
  setResultLayoutAttr(op);
  return success();
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
    Operation *op) {
  return TypeSwitch<Operation *, CompatibilityResult>(op)
      // Trivially true ops
      .Case<func::FuncOp, GenericOp, YieldOp>(
          [&](auto op) { return CompatibilityResult{true, std::nullopt}; })
      // Ops with special rules
      .Case<ReduceOp, VecmatOp>(
          [&](auto op) { return hasCompatibleArgumentLayouts(op); })
      // By default, assume operands must all have the same layout.
      .Default([&](Operation *op) {
        std::optional<AffineMap> firstFoundLayout;

        for (auto &operand : op->getOpOperands()) {
          if (isa<RankedTensorType>(operand.get().getType())) {
            if (!assignedLayouts.contains(operand.get())) {
              // If the operand has no layout, we can't propagate layout
              // information to the result.
              return CompatibilityResult{
                  false, op->emitError("operand has no assigned layout")};
            }
            AffineMap layout = assignedLayouts.at(operand.get());

            if (!firstFoundLayout.has_value()) firstFoundLayout = layout;
            if (layout != firstFoundLayout.value()) {
              return CompatibilityResult{false, std::nullopt};
            }
          }
        }

        return CompatibilityResult{true, std::nullopt};
      });
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
    ReduceOp op) {
  // The arguments of a ReduceOp are the tensor(s) to reduce and the
  // initializer values for the reduction.
  for (const auto &[input, init] : llvm::zip(op.getInputs(), op.getInits())) {
    if (!assignedLayouts.contains(input)) {
      return {false, op->emitError("input tensor has no assigned layout")};
    }
    if (!assignedLayouts.contains(init)) {
      return {false,
              op->emitError("initializer tensor has no assigned layout")};
    }

    AffineMap inputLayout = assignedLayouts.at(input);
    AffineMap initLayout = assignedLayouts.at(init);
    AffineMap reducedInputLayout =
        convertLayoutForReduce(inputLayout, op.getDimensions());

    if (reducedInputLayout != initLayout) {
      return {false, std::nullopt};
    }
  }

  return {true, std::nullopt};
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
    VecmatOp op) {
  // Currently only support secret vectors and plaintext matrices.
  linalg::ContractionOpInterface vecmatOp =
      cast<linalg::ContractionOpInterface>(op.getOperation());
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

void LayoutPropagation::rectifyIncompatibleOperandLayouts(Operation *op) {
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
      note << "\n- Operand: " << operand << "; Layout: " << os.str();
    }
  });

  TypeSwitch<Operation *>(op)
      // Ops with special rules
      .Case<ReduceOp>(
          [&](auto op) { return rectifyIncompatibleOperandLayouts(op); })
      .Default([&](Operation *op) {
        // Default target layout is chosen arbitrarily as the first operand's
        // layout for now. A different pass is responsible for optimizing the
        // placement and mechanics of the layout conversion ops.
        mlir::IRRewriter builder(&getContext());
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
      });
}

void LayoutPropagation::rectifyIncompatibleOperandLayouts(ReduceOp op) {
  mlir::IRRewriter builder(&getContext());
  builder.setInsertionPoint(op);

  for (const auto &[input, init] : llvm::zip(op.getInputs(), op.getInits())) {
    AffineMap inputLayout = assignedLayouts.at(input);
    AffineMap initLayout = assignedLayouts.at(init);
    AffineMap reducedInputLayout =
        convertLayoutForReduce(inputLayout, op.getDimensions());

    if (reducedInputLayout != initLayout) {
      ConvertLayoutOp convertOp = builder.create<ConvertLayoutOp>(
          op->getLoc(), init, AffineMapAttr::get(initLayout),
          AffineMapAttr::get(reducedInputLayout));
      Value toReplace = convertOp.getResult();
      builder.replaceUsesWithIf(init, toReplace, [&](OpOperand &operand) {
        return operand.getOwner() == op;
      });
      assignedLayouts.insert({toReplace, reducedInputLayout});
      setResultLayoutAttr(convertOp);
    }
  }
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
