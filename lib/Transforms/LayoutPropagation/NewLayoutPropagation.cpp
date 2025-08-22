#include "lib/Transforms/LayoutPropagation/NewLayoutPropagation.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/LayoutPropagation/Utils.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"       // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"      // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
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
#include "mlir/include/mlir/Support/WalkResult.h"        // from @llvm-project

#define DEBUG_TYPE "new-layout-propagation"

namespace mlir {
namespace heir {

using linalg::MatvecOp;
using linalg::ReduceOp;
using linalg::VecmatOp;
using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::VarKind;
using secret::GenericOp;
using secret::SecretType;
using secret::YieldOp;
using tensor::CollapseShapeOp;
using tensor::ExpandShapeOp;
using tensor_ext::AssignLayoutOp;
using tensor_ext::ConvertLayoutOp;
using tensor_ext::NewLayoutAttr;

#define GEN_PASS_DEF_NEWLAYOUTPROPAGATION
#include "lib/Transforms/LayoutPropagation/NewLayoutPropagation.h.inc"

namespace {

// The result of a compatibility check for the layouts of an op's operands (cf.
// hasCompatibleArgumentLayouts). If the check fails, the presence of a
// diagnostic signals that the failure is unrecoverable and should cause the
// pass to fail. If the diagnostic is nullopt, then the failure can be
// recovered by rectifyIncompatibleOperandLayouts.
struct CompatibilityResult {
  bool compatible;
  std::optional<InFlightDiagnostic> diag;
};

void visitDebugInfo(Operation* op) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting: " << op->getName() << "\n");
}

void debugAssignLayout(Value value, NewLayoutAttr layout) {
  LLVM_DEBUG(llvm::dbgs() << "Assigning layout " << layout << " to value "
                          << value << "\n");
}

}  // namespace

struct NewLayoutPropagation
    : impl::NewLayoutPropagationBase<NewLayoutPropagation> {
  using NewLayoutPropagationBase::NewLayoutPropagationBase;

  // Top level visit method handling common logic and dispatching to specific
  // visitOperation overloads.
  LogicalResult visitOperation(Operation* op);

  // Op-specific transfer functions
  LogicalResult visitOperation(CollapseShapeOp op);
  LogicalResult visitOperation(ExpandShapeOp op);
  LogicalResult visitOperation(GenericOp op);
  LogicalResult visitOperation(ReduceOp op);
  LogicalResult visitOperation(VecmatOp op);
  LogicalResult visitOperation(MatvecOp op);
  LogicalResult visitOperation(YieldOp op);
  LogicalResult visitOperation(affine::AffineForOp op);
  LogicalResult visitOperation(func::FuncOp op);
  LogicalResult visitOperation(func::ReturnOp op);
  LogicalResult visitOperation(tensor::ExtractOp op);

  // Determine if the operation arguments have compatible layouts for the given
  // op. If the check fails, the CompatibilityResult::compatible field is
  // false. If there is also a diagnostic populated in the result, the failure
  // is unrecoverable.
  CompatibilityResult hasCompatibleArgumentLayouts(Operation* op);

  // Op-specific compatibility functions
  CompatibilityResult hasCompatibleArgumentLayouts(ReduceOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(VecmatOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(MatvecOp op);

  // Insert conversion ops to rectify incompatible operand layouts
  void rectifyIncompatibleOperandLayouts(Operation* op);

  // Op-specific overrides
  void rectifyIncompatibleOperandLayouts(ReduceOp op);

  // Return the default layout for a given type
  FailureOr<NewLayoutAttr> defaultLayoutForType(Type type);
  FailureOr<NewLayoutAttr> defaultLayoutForScalarType(Type type);

  // Create an assign_layout op for the given value, and return the resulting
  // op. The given builder should have its insertion point set before calling.
  FailureOr<AssignLayoutOp> assignDefaultLayoutForOpOperand(
      Operation* op, Value operand, IRRewriter& builder);

  // Helper to pass layouts through generic ops
  void passLayoutThroughOp(Operation* op);

  // Add an op attribute denoting the layouts of the op results. Assumes the
  // assignedLayouts map contains the layout for the result SSA values already.
  void setResultLayoutAttr(Operation* op);

  void runOnOperation() override;

  DenseMap<Value, NewLayoutAttr> assignedLayouts;
  DataFlowSolver* solver;
};

FailureOr<AssignLayoutOp> NewLayoutPropagation::assignDefaultLayoutForOpOperand(
    Operation* op, Value operand, IRRewriter& builder) {
  FailureOr<NewLayoutAttr> layout = defaultLayoutForType(operand.getType());
  if (failed(layout)) {
    return failure();
  }
  NewLayoutAttr layoutAttr = layout.value();
  AssignLayoutOp assignLayoutOp =
      AssignLayoutOp::create(builder, op->getLoc(), operand, layoutAttr);
  setAttributeAssociatedWith(assignLayoutOp.getResult(),
                             tensor_ext::TensorExtDialect::kLayoutAttrName,
                             layoutAttr);
  Value toReplace = assignLayoutOp.getResult();
  // This may create duplicate layout assignment ops, and we expect CSE
  // to later clean them up. Otherwise we risk replacing a use of the
  // cleartext value in some other context.
  builder.replaceUsesWithIf(operand, toReplace, [&](OpOperand& otherOperand) {
    return otherOperand.getOwner() == op;
  });
  debugAssignLayout(toReplace, layoutAttr);
  assignedLayouts.insert({toReplace, layoutAttr});

  return assignLayoutOp;
}

LogicalResult NewLayoutPropagation::visitOperation(Operation* op) {
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
  for (Value operand : op->getOperands()) {
    if (!assignedLayouts.contains(operand)) {
      if (isa<IndexType>(operand.getType())) {
        // TODO (#1929): Index types sometimes do need to be layout-matched
        // (e.g., arith.cmpi) but sometimes do not (e.g., tensor_ext.rotate). We
        // should probably have an op interface to determine which operands need
        // a layout?
        // For now, here's a hack to make it work for the arith.cmpi case:
        if (!isa<arith::CmpIOp, arith::CmpFOp>(op)) continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "operand has no layout: " << operand << "\n");
      mlir::IRRewriter builder(&getContext());
      builder.setInsertionPoint(op);
      if (failed(assignDefaultLayoutForOpOperand(op, operand, builder)))
        return op->emitError()
               << "Failed to assign default layout to operand " << operand;
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

  return TypeSwitch<Operation*, LogicalResult>(op)
      // func ops
      .Case<func::FuncOp, func::ReturnOp>(
          [&](auto op) { return visitOperation(op); })
      // secret ops
      .Case<GenericOp, YieldOp>([&](auto op) { return visitOperation(op); })
      // linalg ops
      .Case<MatvecOp, VecmatOp, ReduceOp>(
          [&](auto op) { return visitOperation(op); })
      // affine ops
      .Case<affine::AffineForOp>([&](auto op) { return visitOperation(op); })
      // tensor ops
      .Case<tensor::ExtractOp>([&](auto op) { return visitOperation(op); })
      .Case<tensor::ExtractSliceOp, tensor::InsertSliceOp>([&](auto op) {
        // TODO(#2028): Support tensor.extract_slice and tensor.insert_slice in
        // layout.
        return op->emitError()
               << "Layout propagation not supported for this op";
      })
      // tensor ops
      .Case<CollapseShapeOp, ExpandShapeOp>(
          [&](auto op) { return visitOperation(op); })
      // AddI, AddF, mgmt.* all pass the layout through unchanged.
      .Default([&](Operation* op) {
        passLayoutThroughOp(op);
        return success();
      });
}

LogicalResult NewLayoutPropagation::visitOperation(func::FuncOp op) {
  // Set a default value for each secret argument
  for (Value arg : op.getArguments()) {
    if (!isSecret(arg, solver)) {
      // Cleartext arguments don't get layouts, they are later given
      // assign_layout ops and materialized to plaintexts server-side.
      continue;
    }
    FailureOr<NewLayoutAttr> layout = defaultLayoutForType(arg.getType());
    if (failed(layout)) {
      return op->emitOpError()
             << "Failed to assign default layout to func argument " << arg;
    }
    debugAssignLayout(arg, layout.value());
    assignedLayouts.insert({arg, layout.value()});
    setAttributeAssociatedWith(
        arg, tensor_ext::TensorExtDialect::kLayoutAttrName, layout.value());
  }

  // Func result attrs are handled by the ReturnOp
  return success();
}

LogicalResult NewLayoutPropagation::visitOperation(func::ReturnOp op) {
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  for (OpOperand& operand : op->getOpOperands()) {
    if (!assignedLayouts.contains(operand.get())) {
      if (isSecret(operand.get(), solver)) {
        return op->emitError("secret return value has no assigned layout");
      }

      // It needs no layout.
      continue;
    }
    NewLayoutAttr layout = assignedLayouts.at(operand.get());
    func.setResultAttr(operand.getOperandNumber(),
                       tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
  }
  return success();
}

LogicalResult NewLayoutPropagation::visitOperation(GenericOp op) {
  // Every block argument has the same layout as its corresponding operand.
  for (OpOperand& operand : op->getOpOperands()) {
    if (!assignedLayouts.contains(operand.get())) {
      // Assume it is not a tensor type and doesn't need a layout.
      continue;
    }
    NewLayoutAttr layout = assignedLayouts.at(operand.get());
    BlockArgument blockArg =
        op.getRegion().getArgument(operand.getOperandNumber());
    assignedLayouts.insert({blockArg, layout});
    setAttributeAssociatedWith(
        blockArg, tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
    debugAssignLayout(blockArg, layout);
  }
  // The layout of the result of the generic op is handled when the YieldOp is
  // visited.
  return success();
}

LogicalResult NewLayoutPropagation::visitOperation(YieldOp op) {
  // The results of the generic op has the same layouts as the yielded values
  GenericOp generic = op->getParentOfType<GenericOp>();
  for (OpOperand& operand : op->getOpOperands()) {
    Type operandType = operand.get().getType();
    if (!assignedLayouts.contains(operand.get())) {
      // If it's a tensor type, it may be something like a tensor.empty()
      // that would not be assigned a layout earlier in the walk, because
      // it does not depend on any secret information. In this case, use the
      // default layout.
      LLVM_DEBUG(llvm::dbgs() << "No layout assigned to operand "
                              << operand.get() << ", using default layout\n");
      if (isa<RankedTensorType>(operandType)) {
        FailureOr<NewLayoutAttr> layout = defaultLayoutForType(operandType);
        if (failed(layout)) {
          return failure();
        }
        NewLayoutAttr layoutAttr = layout.value();
        debugAssignLayout(operand.get(), layoutAttr);
        assignedLayouts.insert({operand.get(), layoutAttr});
      } else {
        // Assume it is not a tensor type and doesn't need a layout.
        continue;
      }
    }
    NewLayoutAttr layout = assignedLayouts.at(operand.get());
    Value result = generic.getResult(operand.getOperandNumber());
    assignedLayouts.insert({result, layout});
    debugAssignLayout(result, layout);
    setAttributeAssociatedWith(
        result, tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
  }
  return success();
}

LogicalResult NewLayoutPropagation::visitOperation(CollapseShapeOp op) {
  // Only support rank-reduced types for now, i.e., where the collapsed
  // shape only removes static dimensions of size 1.
  SliceVerificationResult res =
      isRankReducedType(op.getSrcType(), op.getResultType());
  if (res != SliceVerificationResult::Success)
    return op->emitError(
        "Only rank-reduced types are supported for CollapseShapeOp");

  // Set to zero and eliminate the dropped dimensions.
  auto tensor = op.getSrc();
  NewLayoutAttr inputLayout = assignedLayouts.at(tensor);
  IntegerRelation relation = inputLayout.getIntegerRelation();
  std::unique_ptr<IntegerRelation> clonedRelation = relation.clone();

  for (Attribute associationGroup : op.getReassociation()) {
    auto associationArray = dyn_cast<ArrayAttr>(associationGroup).getValue();
    // a single-entry association group is a no-op
    if (associationArray.size() == 1) {
      continue;
    }
    for (Attribute association : associationArray) {
      int64_t reassocDim = cast<IntegerAttr>(association).getInt();
      if (op.getSrcType().getShape()[reassocDim] == 1) {
        // Drop this unit dimension
        clonedRelation->setAndEliminate(reassocDim, 0);
      }
    }
  }

  MLIRContext* ctx = &getContext();
  NewLayoutAttr resultLayoutAttr =
      NewLayoutAttr::getFromIntegerRelation(ctx, *clonedRelation);

  assignedLayouts.insert({op.getResult(), resultLayoutAttr});
  setResultLayoutAttr(op);
  debugAssignLayout(op.getResult(), resultLayoutAttr);
  return success();
}

LogicalResult NewLayoutPropagation::visitOperation(ExpandShapeOp op) {
  // Only support rank-reduced types for now, i.e., where the expanded shape
  // only adds static dimensions of size 1.
  SliceVerificationResult res =
      isRankReducedType(op.getResultType(), op.getSrcType());
  if (res != SliceVerificationResult::Success)
    return op->emitError(
        "Only rank-reduced types are supported for ExpandShapeOp");

  auto tensor = op.getSrc();
  NewLayoutAttr inputLayout = assignedLayouts.at(tensor);
  IntegerRelation relation = inputLayout.getIntegerRelation();
  std::unique_ptr<IntegerRelation> clonedRelation = relation.clone();

  // tensor indices correspond to layout dimensions, and adding a dimension of
  // size 1 has no effect on the affine map expressions, so all we're doing is
  // adding new dimensions for each reassociation group index corresponding to
  // an output dimension of size 1. Mainly we have to ensure that the
  // dimension we're adding is in the correct index of the integer relations
  // domain variable list.
  int oldDim = 0;
  DenseMap<AffineExpr, AffineExpr> oldDimsToNewDims;
  for (Attribute associationGroup : op.getReassociation()) {
    auto associationArray = dyn_cast<ArrayAttr>(associationGroup).getValue();
    // a single-entry association group is a no-op
    if (associationArray.size() == 1) {
      ++oldDim;
      continue;
    }

    for (Attribute association : associationArray) {
      int64_t reassocDim = cast<IntegerAttr>(association).getInt();
      if (op.getResultType().getShape()[reassocDim] > 1) {
        ++oldDim;
      } else {
        // A new dimension of size 1 is being added, so add a new domain
        // variable v with 0 <= v < 1.
        auto newDimIndex = clonedRelation->insertVar(VarKind::Domain, oldDim);
        clonedRelation->addBound(BoundType::LB, newDimIndex, 0);
        clonedRelation->addBound(BoundType::UB, newDimIndex, 0);
        ++oldDim;
      }
    }
  }

  MLIRContext* ctx = &getContext();
  NewLayoutAttr resultLayoutAttr =
      NewLayoutAttr::getFromIntegerRelation(ctx, *clonedRelation);

  assignedLayouts.insert({op.getResult(), resultLayoutAttr});
  setResultLayoutAttr(op);
  debugAssignLayout(op.getResult(), resultLayoutAttr);
  return success();
}

LogicalResult NewLayoutPropagation::visitOperation(VecmatOp op) {
  auto vecmatOp = cast<linalg::ContractionOpInterface>(*op);
  auto vec = vecmatOp.lhs();

  // The matrix has no assigned layout because it is assumed to be
  // plaintext/static (this is intended to be enforced by
  // hasCompatibleArgumentLayouts).
  NewLayoutAttr vecLayout = assignedLayouts.at(vec);

  // Always one result, and it's a vector with the same layout
  // as the input vector
  auto result = vecmatOp->getResult(0);
  NewLayoutAttr resultLayout = vecLayout;

  assignedLayouts.insert({result, resultLayout});
  setResultLayoutAttr(op);
  debugAssignLayout(result, resultLayout);
  return success();
}

LogicalResult NewLayoutPropagation::visitOperation(MatvecOp op) {
  auto matvecOp = cast<linalg::ContractionOpInterface>(*op);
  auto matrix = matvecOp.lhs();
  auto matrixType = cast<RankedTensorType>(matrix.getType());

  // TODO(#1597): a layout optimizer should really be selecting the diagonal
  // layout instead of this pass.

  NewLayoutAttr matrixLayout = assignedLayouts.at(matrix);
  // The Halevi-Shoup kernel (all we support at this time) requires one
  // ciphertext per matrix row.
  int64_t numCiphertexts = matrixType.getShape()[0];
  RankedTensorType ciphertextSemanticShape = RankedTensorType::get(
      {numCiphertexts, ciphertextSize}, matrixType.getElementType());
  if (!isRelationSquatDiagonal(matrixType, ciphertextSemanticShape,
                               matrixLayout.getIntegerRelation())) {
    // Insert a layout conversion op to make the matrix layout squat diagonal
    MLIRContext* ctx = &getContext();
    mlir::IRRewriter builder(ctx);
    builder.setInsertionPoint(op);

    IntegerRelation diagonalRelation =
        getDiagonalLayoutRelation(matrixType, ciphertextSemanticShape);
    NewLayoutAttr squatDiagonalLayoutAttr =
        NewLayoutAttr::getFromIntegerRelation(ctx, diagonalRelation);

    ConvertLayoutOp convertLayoutOp = ConvertLayoutOp::create(
        builder, op->getLoc(), matrix, matrixLayout, squatDiagonalLayoutAttr);
    convertLayoutOp->setAttr(tensor_ext::TensorExtDialect::kLayoutAttrName,
                             squatDiagonalLayoutAttr);
    Value toReplace = convertLayoutOp.getResult();
    builder.replaceUsesWithIf(matrix, toReplace, [&](OpOperand& operand) {
      return operand.getOwner() == op;
    });
    debugAssignLayout(toReplace, squatDiagonalLayoutAttr);
    assignedLayouts.insert({toReplace, squatDiagonalLayoutAttr});
    matrix = toReplace;
  }

  // Always one result, and for the kernels we have right now it's always a
  // row-major replicated vector. Since the matrix may be rectangular, the
  // output layout may have different alignment from the input layout.
  auto result = matvecOp->getResult(0);
  RankedTensorType outputType = cast<RankedTensorType>(result.getType());
  FailureOr<NewLayoutAttr> outputLayoutResult =
      defaultLayoutForType(outputType);
  if (failed(outputLayoutResult)) {
    return failure();
  }
  NewLayoutAttr resultLayout = outputLayoutResult.value();

  assignedLayouts.insert({result, resultLayout});
  setResultLayoutAttr(op);
  debugAssignLayout(result, resultLayout);
  return success();
}

LogicalResult NewLayoutPropagation::visitOperation(ReduceOp op) {
  for (const auto& [tensor, result] :
       llvm::zip(op.getInputs(), op.getResults())) {
    NewLayoutAttr resultLayout =
        convertLayoutForReduce(assignedLayouts.at(tensor), op.getDimensions());
    assignedLayouts.insert({result, resultLayout});
    debugAssignLayout(result, resultLayout);
  }
  setResultLayoutAttr(op);
  return success();
}

LogicalResult NewLayoutPropagation::visitOperation(affine::AffineForOp op) {
  // Transfer the layout of the inits to the region iter args
  for (const auto& [init, iterArg, result] :
       llvm::zip(op.getInits(), op.getRegionIterArgs(), op.getResults())) {
    NewLayoutAttr layout;

    // The init may not have a layout if it is a non-secret initial value. In
    // that case we assign it a default layout.
    if (!assignedLayouts.contains(init)) {
      mlir::IRRewriter builder(&getContext());
      builder.setInsertionPoint(op);
      auto res = assignDefaultLayoutForOpOperand(op, init, builder);
      if (failed(res)) {
        return op->emitError()
               << "Failed to assign default layout to init " << init;
      }
      layout = cast<NewLayoutAttr>(res.value().getLayout());
    } else {
      layout = assignedLayouts.at(init);
    }
    assignedLayouts.insert({iterArg, layout});
    setAttributeAssociatedWith(
        iterArg, tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
    debugAssignLayout(iterArg, layout);

    // The result of an AffineForOp also has the same type as its iter args.
    assignedLayouts.insert({result, layout});
    debugAssignLayout(result, layout);
  }

  setResultLayoutAttr(op);
  return success();
}

LogicalResult NewLayoutPropagation::visitOperation(tensor::ExtractOp op) {
  // Use the default scalar layout for the extracted scalar.
  FailureOr<NewLayoutAttr> scalarLayout =
      defaultLayoutForScalarType(op.getResult().getType());
  if (failed(scalarLayout)) {
    return op->emitError("Failed to get scalar layout for extract result");
  }

  LLVM_DEBUG(llvm::dbgs() << "Assigning scalar layout " << scalarLayout.value()
                          << " to value " << op.getResult() << "\n");

  Value result = op.getResult();
  NewLayoutAttr resultLayout = scalarLayout.value();
  assignedLayouts.insert({result, resultLayout});
  debugAssignLayout(result, resultLayout);
  setResultLayoutAttr(op);

  return success();
}

CompatibilityResult NewLayoutPropagation::hasCompatibleArgumentLayouts(
    Operation* op) {
  return TypeSwitch<Operation*, CompatibilityResult>(op)
      // Trivially true ops
      .Case<func::FuncOp, GenericOp, YieldOp, affine::AffineForOp,
            affine::AffineYieldOp>(
          [&](auto op) { return CompatibilityResult{true, std::nullopt}; })
      // Ops with special rules
      .Case<ReduceOp, MatvecOp, VecmatOp>(
          [&](auto op) { return hasCompatibleArgumentLayouts(op); })
      // By default, assume operands must all have the same layout.
      .Default([&](Operation* op) {
        std::optional<NewLayoutAttr> firstFoundLayout;

        for (auto& operand : op->getOpOperands()) {
          if (isa<RankedTensorType>(operand.get().getType())) {
            if (!assignedLayouts.contains(operand.get())) {
              // If the operand has no layout, we can't propagate layout
              // information to the result.
              return CompatibilityResult{
                  false, op->emitError("operand has no assigned layout")};
            }
            NewLayoutAttr layout = assignedLayouts.at(operand.get());

            if (!firstFoundLayout.has_value()) firstFoundLayout = layout;
            if (layout != firstFoundLayout.value()) {
              return CompatibilityResult{false, std::nullopt};
            }
          }
        }

        return CompatibilityResult{true, std::nullopt};
      });
}

CompatibilityResult NewLayoutPropagation::hasCompatibleArgumentLayouts(
    ReduceOp op) {
  // The arguments of a ReduceOp are the tensor(s) to reduce and the
  // initializer values for the reduction.
  for (const auto& [input, init] : llvm::zip(op.getInputs(), op.getInits())) {
    if (!assignedLayouts.contains(input)) {
      return {false, op->emitError("input tensor has no assigned layout")};
    }
    if (!assignedLayouts.contains(init)) {
      return {false,
              op->emitError("initializer tensor has no assigned layout")};
    }

    NewLayoutAttr inputLayout = assignedLayouts.at(input);
    NewLayoutAttr initLayout = assignedLayouts.at(init);
    NewLayoutAttr reducedInputLayout =
        convertLayoutForReduce(inputLayout, op.getDimensions());

    if (reducedInputLayout != initLayout) {
      return {false, std::nullopt};
    }
  }

  return {true, std::nullopt};
}

CompatibilityResult NewLayoutPropagation::hasCompatibleArgumentLayouts(
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

CompatibilityResult NewLayoutPropagation::hasCompatibleArgumentLayouts(
    MatvecOp op) {
  // Currently only support secret vectors and plaintext matrices.
  linalg::ContractionOpInterface matvecOp =
      cast<linalg::ContractionOpInterface>(op.getOperation());
  Value vec = matvecOp.rhs();
  Value mat = matvecOp.lhs();
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

void NewLayoutPropagation::rectifyIncompatibleOperandLayouts(Operation* op) {
  LLVM_DEBUG({
    auto diag = op->emitRemark() << "Inserting layout conversion op due to "
                                    "disagreeing operand layouts";
    auto& note = diag.attachNote();
    for (auto operand : op->getOperands()) {
      NewLayoutAttr operandLayout;
      if (assignedLayouts.contains(operand))
        operandLayout = assignedLayouts.at(operand);
      note << "\n- Operand: " << operand << "; Layout: " << operandLayout;
    }
  });

  TypeSwitch<Operation*>(op)
      // Ops with special rules
      .Case<ReduceOp>(
          [&](auto op) { return rectifyIncompatibleOperandLayouts(op); })
      .Default([&](Operation* op) {
        // Default target layout is chosen arbitrarily as the first operand's
        // layout for now. A different pass is responsible for optimizing the
        // placement and mechanics of the layout conversion ops.
        mlir::IRRewriter builder(&getContext());
        const auto it = llvm::find_if(op->getOperands(), [this](Value pair) {
          return assignedLayouts.contains(pair);
        });
        NewLayoutAttr targetLayout = assignedLayouts.at(*it);

        for (auto& opOperand : op->getOpOperands()) {
          if (!assignedLayouts.contains(opOperand.get())) continue;
          NewLayoutAttr sourceLayout = assignedLayouts.at(opOperand.get());

          if (sourceLayout != targetLayout) {
            builder.setInsertionPoint(op);
            ConvertLayoutOp convertOp =
                ConvertLayoutOp::create(builder, op->getLoc(), opOperand.get(),
                                        sourceLayout, targetLayout);

            // Layout of the result is the same as the target layout of the
            // conversion. Mostly this is done for consistency: all ops have
            // an attribute describing the layout of their results.
            OpBuilder builder(&getContext());
            assignedLayouts.insert({convertOp.getResult(), targetLayout});
            setResultLayoutAttr(convertOp);
            op->setOperand(opOperand.getOperandNumber(), convertOp.getResult());
          }
        }
      });
}

void NewLayoutPropagation::rectifyIncompatibleOperandLayouts(ReduceOp op) {
  mlir::IRRewriter builder(&getContext());
  builder.setInsertionPoint(op);

  for (const auto& [input, init] : llvm::zip(op.getInputs(), op.getInits())) {
    NewLayoutAttr inputLayout = assignedLayouts.at(input);
    NewLayoutAttr initLayout = assignedLayouts.at(init);
    NewLayoutAttr reducedInputLayout =
        convertLayoutForReduce(inputLayout, op.getDimensions());

    if (reducedInputLayout != initLayout) {
      ConvertLayoutOp convertOp = ConvertLayoutOp::create(
          builder, op->getLoc(), init, initLayout, reducedInputLayout);
      Value toReplace = convertOp.getResult();
      builder.replaceUsesWithIf(init, toReplace, [&](OpOperand& operand) {
        return operand.getOwner() == op;
      });
      assignedLayouts.insert({toReplace, reducedInputLayout});
      setResultLayoutAttr(convertOp);
    }
  }
}

void NewLayoutPropagation::passLayoutThroughOp(Operation* op) {
  // All inputs have the same layout, so just propagate it to all results
  NewLayoutAttr layout = assignedLayouts.at(op->getOperand(0));
  for (Value result : op->getResults()) {
    assignedLayouts.insert({result, layout});
    debugAssignLayout(result, layout);
  }
  setResultLayoutAttr(op);
}

FailureOr<NewLayoutAttr> NewLayoutPropagation::defaultLayoutForScalarType(
    Type scalarType) {
  // FIXME: Does a scalar have domainSize=0? And should the output type be
  // tensor<1 x ciphertextSize x type> or tensor<ciphertextSize x type>?
  auto* ctx = scalarType.getContext();
  OpBuilder builder(ctx);

  // Only support Int/Index/Float scalars,
  // fail for any other random type that might make it here.
  if (!isa<FloatType, IntegerType, IndexType>(scalarType)) {
    return failure();
  }

  std::string relationStr = llvm::formatv(
      "{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= {0} }", ciphertextSize - 1);
  return NewLayoutAttr::get(ctx, relationStr);
}

FailureOr<NewLayoutAttr> NewLayoutPropagation::defaultLayoutForType(Type type) {
  Type ty = type;
  if (SecretType secretType = dyn_cast<SecretType>(type)) {
    ty = secretType.getValueType();
  }

  RankedTensorType tensorType = dyn_cast<RankedTensorType>(ty);
  if (!tensorType) {
    return defaultLayoutForScalarType(ty);
  }

  // By default, each tensor is laid out in row-major order. The slots will be
  // repeated each N elements where N is the next power of two of the total
  // number of data elements.
  LLVM_DEBUG(llvm::dbgs() << "getting row-major layout map for type="
                          << tensorType << "\n");
  IntegerRelation relation =
      getRowMajorLayoutRelation(tensorType, ciphertextSize);
  return NewLayoutAttr::getFromIntegerRelation(tensorType.getContext(),
                                               relation);
}

void NewLayoutPropagation::setResultLayoutAttr(Operation* op) {
  OpBuilder builder(&getContext());
  SmallVector<Attribute> resultLayouts = llvm::map_to_vector(
      op->getResults(),
      [&](Value result) -> Attribute { return assignedLayouts.at(result); });

  if (op->getNumResults() == 1) {
    op->setAttr(tensor_ext::TensorExtDialect::kLayoutAttrName,
                resultLayouts.front());
    return;
  }
  op->setAttr(tensor_ext::TensorExtDialect::kLayoutAttrName,
              builder.getArrayAttr(resultLayouts));
}

void NewLayoutPropagation::runOnOperation() {
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
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
      getOperation()->walk<WalkOrder::PreOrder>([&](Operation* op) {
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
