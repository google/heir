#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/TensorExt/Transforms/Patterns.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Transforms/LayoutPropagation/Utils.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/Hoisting.h"
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
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "layout-propagation"

namespace mlir {
namespace heir {

using linalg::Conv2DOp;
using linalg::MatmulOp;
using linalg::MatvecOp;
using linalg::ReduceOp;
using linalg::VecmatOp;
using presburger::IntegerRelation;
using secret::GenericOp;
using secret::SecretType;
using secret::YieldOp;
using tensor::CollapseShapeOp;
using tensor::ExpandShapeOp;
using tensor::InsertOp;
using tensor_ext::AssignLayoutOp;
using tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;

#define GEN_PASS_DEF_LAYOUTPROPAGATION
#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h.inc"

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

void debugAssignLayout(Value value, LayoutAttr layout) {
  LLVM_DEBUG(llvm::dbgs() << "Assigning layout " << layout << " to value "
                          << value << "\n");
}

std::pair<Value, LayoutAttr> convertToLayout(
    MLIRContext* ctx, mlir::IRRewriter& builder, Operation* op, Value value,
    LayoutAttr oldLayout, const IntegerRelation& newRelation) {
  builder.setInsertionPoint(op);
  LayoutAttr layoutAttr = LayoutAttr::getFromIntegerRelation(ctx, newRelation);
  ConvertLayoutOp convertLayoutOp = ConvertLayoutOp::create(
      builder, op->getLoc(), value, oldLayout, layoutAttr);
  convertLayoutOp->setAttr(tensor_ext::TensorExtDialect::kLayoutAttrName,
                           layoutAttr);
  Value toReplace = convertLayoutOp.getResult();
  builder.replaceUsesWithIf(value, toReplace, [&](OpOperand& operand) {
    return operand.getOwner() == op;
  });
  return std::make_pair(toReplace, layoutAttr);
}

}  // namespace

struct LayoutPropagation : impl::LayoutPropagationBase<LayoutPropagation> {
  using LayoutPropagationBase::LayoutPropagationBase;

  // Top level visit method handling common logic and dispatching to specific
  // visitOperation overloads.
  LogicalResult visitOperation(Operation* op);

  // Op-specific transfer functions
  LogicalResult visitOperation(CollapseShapeOp op);
  LogicalResult visitOperation(ExpandShapeOp op);
  LogicalResult visitOperation(GenericOp op);
  LogicalResult visitOperation(ReduceOp op);
  LogicalResult visitOperation(Conv2DOp op);
  LogicalResult visitOperation(VecmatOp op);
  LogicalResult visitOperation(MatvecOp op);
  LogicalResult visitOperation(MatmulOp op);
  LogicalResult visitOperation(YieldOp op);
  LogicalResult visitOperation(affine::AffineForOp op);
  LogicalResult visitOperation(func::FuncOp op);
  LogicalResult visitOperation(func::ReturnOp op);
  LogicalResult visitOperation(tensor::ExtractOp op);
  LogicalResult visitOperation(tensor::InsertOp op);
  LogicalResult visitOperation(tensor::InsertSliceOp op);
  LogicalResult visitOperation(tensor::ExtractSliceOp op);

  // Determine if the operation arguments have compatible layouts for the
  // given op. If the check fails, the CompatibilityResult::compatible field
  // is false. If there is also a diagnostic populated in the result, the
  // failure is unrecoverable.
  CompatibilityResult hasCompatibleArgumentLayouts(Operation* op);

  // Op-specific compatibility functions
  CompatibilityResult hasCompatibleArgumentLayouts(Conv2DOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(ReduceOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(VecmatOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(MatvecOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(MatmulOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(tensor::InsertSliceOp op);

  // Insert conversion ops to rectify incompatible operand layouts
  void rectifyIncompatibleOperandLayouts(Operation* op);

  // Op-specific overrides
  void rectifyIncompatibleOperandLayouts(ReduceOp op);
  void rectifyIncompatibleOperandLayouts(tensor::InsertOp op);
  void rectifyIncompatibleOperandLayouts(tensor::InsertSliceOp op);

  // Return the default layout for a given type
  FailureOr<LayoutAttr> defaultLayoutForType(Type type);
  FailureOr<LayoutAttr> defaultLayoutForScalarType(Type type);

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

  DenseMap<Value, LayoutAttr> assignedLayouts;
  DataFlowSolver* solver;
};

FailureOr<AssignLayoutOp> LayoutPropagation::assignDefaultLayoutForOpOperand(
    Operation* op, Value operand, IRRewriter& builder) {
  FailureOr<LayoutAttr> layout = defaultLayoutForType(operand.getType());
  if (failed(layout)) {
    return failure();
  }
  LayoutAttr layoutAttr = layout.value();
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

LogicalResult LayoutPropagation::visitOperation(Operation* op) {
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
  for (OpOperand& opOperand : op->getOpOperands()) {
    Value operand = opOperand.get();
    if (!assignedLayouts.contains(operand)) {
      // Some operations may function properly with operands that have no
      // layout, e.g., a tensor_ext.rotate op doesn't need a layout for the
      // shift operand, and a tensor.insert op has a kernel that can work with a
      // cleartext scalar operand (and is, in fact, more efficient than when the
      // scalar operand is packed).
      if (auto layoutReqIface =
              dyn_cast<OperandLayoutRequirementOpInterface>(op)) {
        bool secretness = isSecret(opOperand.get(), solver);
        if (!layoutReqIface.operandRequiresLayout(opOperand.getOperandNumber(),
                                                  secretness)) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "OperandLayoutRequirementOpInterface ensures us that operand "
              << opOperand.getOperandNumber() << " of op " << op->getName()
              << " does not need a layout.\n");
          continue;
        }
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
      .Case<MatvecOp, VecmatOp, ReduceOp, MatmulOp, Conv2DOp>(
          [&](auto op) { return visitOperation(op); })
      // affine ops
      .Case<affine::AffineForOp>([&](auto op) { return visitOperation(op); })
      // tensor ops
      .Case<tensor::ExtractOp, tensor::InsertOp, tensor::InsertSliceOp,
            tensor::ExtractSliceOp, CollapseShapeOp, ExpandShapeOp>(
          [&](auto op) { return visitOperation(op); })
      // AddI, AddF, mgmt.* all pass the layout through unchanged.
      .Default([&](Operation* op) {
        passLayoutThroughOp(op);
        return success();
      });
}

LogicalResult LayoutPropagation::visitOperation(func::FuncOp op) {
  // Set a default value for each secret argument
  for (Value arg : op.getArguments()) {
    if (!isSecret(arg, solver)) {
      // Cleartext arguments don't get layouts, they are later given
      // assign_layout ops and materialized to plaintexts server-side.
      continue;
    }
    FailureOr<LayoutAttr> layout = defaultLayoutForType(arg.getType());
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

LogicalResult LayoutPropagation::visitOperation(func::ReturnOp op) {
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  for (OpOperand& operand : op->getOpOperands()) {
    if (!assignedLayouts.contains(operand.get())) {
      if (isSecret(operand.get(), solver)) {
        return op->emitError("secret return value has no assigned layout");
      }

      // It needs no layout.
      continue;
    }
    LayoutAttr layout = assignedLayouts.at(operand.get());
    func.setResultAttr(operand.getOperandNumber(),
                       tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
  }
  return success();
}

LogicalResult LayoutPropagation::visitOperation(GenericOp op) {
  // Every block argument has the same layout as its corresponding operand.
  for (OpOperand& operand : op->getOpOperands()) {
    if (!assignedLayouts.contains(operand.get())) {
      // Assume it is not a tensor type and doesn't need a layout.
      continue;
    }
    LayoutAttr layout = assignedLayouts.at(operand.get());
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

LogicalResult LayoutPropagation::visitOperation(YieldOp op) {
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
        FailureOr<LayoutAttr> layout = defaultLayoutForType(operandType);
        if (failed(layout)) {
          return failure();
        }
        LayoutAttr layoutAttr = layout.value();
        debugAssignLayout(operand.get(), layoutAttr);
        assignedLayouts.insert({operand.get(), layoutAttr});
      } else {
        // Assume it is not a tensor type and doesn't need a layout.
        continue;
      }
    }
    LayoutAttr layout = assignedLayouts.at(operand.get());
    Value result = generic.getResult(operand.getOperandNumber());
    assignedLayouts.insert({result, layout});
    debugAssignLayout(result, layout);
    setAttributeAssociatedWith(
        result, tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
  }
  return success();
}

LogicalResult LayoutPropagation::visitOperation(CollapseShapeOp op) {
  // Reassociate the dimensions to collapse. For e.g. if we collapse d0 x d1 x
  // d2 to d' then we need to construct the row-major relation from (d0, d1, d2)
  // -> d'. Then the relation on the collapsed tensor will be the composition
  // from d' -> (d0, d1, d2) -> (ct, slot).
  auto rowMajorRelation = getCollapsedRelation(
      op.getSrcType(), op.getResultType(), op.getReassociationIndices());

  auto tensor = op.getSrc();
  LayoutAttr inputLayout = assignedLayouts.at(tensor);
  IntegerRelation relation = inputLayout.getIntegerRelation();

  relation.applyDomain(rowMajorRelation);
  MLIRContext* ctx = &getContext();
  LayoutAttr resultLayoutAttr =
      LayoutAttr::getFromIntegerRelation(ctx, relation);

  assignedLayouts.insert({op.getResult(), resultLayoutAttr});
  setResultLayoutAttr(op);
  debugAssignLayout(op.getResult(), resultLayoutAttr);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(ExpandShapeOp op) {
  // Only support rank-reduced types for now, i.e., where the expanded shape
  // only adds static dimensions of size 1.
  SliceVerificationResult res =
      isRankReducedType(op.getResultType(), op.getSrcType());
  if (res != SliceVerificationResult::Success)
    return op->emitError(
        "Only rank-reduced types are supported for ExpandShapeOp");

  auto tensor = op.getSrc();
  LayoutAttr inputLayout = assignedLayouts.at(tensor);
  IntegerRelation relation = inputLayout.getIntegerRelation();
  IntegerRelation expandedRelation = expandDimensions(
      relation, op.getResultType(), op.getReassociationIndices());

  MLIRContext* ctx = &getContext();
  LayoutAttr resultLayoutAttr =
      LayoutAttr::getFromIntegerRelation(ctx, expandedRelation);

  assignedLayouts.insert({op.getResult(), resultLayoutAttr});
  setResultLayoutAttr(op);
  debugAssignLayout(op.getResult(), resultLayoutAttr);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(VecmatOp op) {
  auto vecmatOp = cast<linalg::ContractionOpInterface>(*op);
  auto vec = vecmatOp.lhs();
  auto vecType = cast<RankedTensorType>(vec.getType());
  auto matrix = vecmatOp.rhs();
  if (!isSecret(vec, solver) || isSecret(matrix, solver)) {
    return failure();
  }

  MLIRContext* ctx = &getContext();
  mlir::IRRewriter builder(ctx);

  if (vecType.getDimSize(0) > ciphertextSize) {
    return op->emitError() << "Vector must fit into a single ciphertext";
  }

  // Assign a row major layout to the input vec.
  LayoutAttr vecLayout = assignedLayouts.at(vec);
  if (!isRelationRowMajor(vecType, ciphertextSize,
                          vecLayout.getIntegerRelation())) {
    // Insert a layout conversion op to make the vec layout per-row
    auto [toReplace, newVecLayoutAttr] =
        convertToLayout(ctx, builder, op, vec, vecLayout,
                        getPerRowLayoutRelation(vecType, ciphertextSize));
    debugAssignLayout(toReplace, newVecLayoutAttr);
    assignedLayouts.insert({toReplace, newVecLayoutAttr});
    vec = toReplace;
  }

  // Assign or change the filter matrix to a diagonal layout. For a vecmat
  // operation, we need the diagonals to be extracted from the transpose.
  // So we get the regular diagonal layout for the transpose matrix type, and
  // then swap the dimensions.
  auto matrixType = cast<RankedTensorType>(matrix.getType());
  auto matrixTransposeType = RankedTensorType::get(
      {matrixType.getDimSize(1), matrixType.getDimSize(0)},
      matrixType.getElementType());
  LayoutAttr matrixLayout = assignedLayouts.at(matrix);
  IntegerRelation matrixLayoutRelation = matrixLayout.getIntegerRelation();
  auto domainOffset = matrixLayout.getIntegerRelation().getVarKindOffset(
      presburger::VarKind::Domain);
  auto clonedMatrixRelation = matrixLayoutRelation.clone();
  clonedMatrixRelation->swapVar(domainOffset, domainOffset + 1);
  if (!isRelationSquatDiagonal(matrixTransposeType, ciphertextSize,
                               *clonedMatrixRelation)) {
    // Insert a layout conversion op to make the matrix layout squat diagonal
    auto [toReplace, newMatrixLayoutAttr] = convertToLayout(
        ctx, builder, op, matrix, matrixLayout,
        getDiagonalLayoutRelation(matrixTransposeType, ciphertextSize));
    debugAssignLayout(toReplace, newMatrixLayoutAttr);
    assignedLayouts.insert({toReplace, newMatrixLayoutAttr});
    matrix = toReplace;
  }

  // The output has the same per-row layout as the input matrix.
  auto result = vecmatOp->getResult(0);
  RankedTensorType outputType = cast<RankedTensorType>(result.getType());
  IntegerRelation outputLayoutResult =
      getRowMajorLayoutRelation(outputType, ciphertextSize);
  LayoutAttr outputLayoutAttr =
      LayoutAttr::getFromIntegerRelation(ctx, outputLayoutResult);

  assignedLayouts.insert({result, outputLayoutAttr});
  setResultLayoutAttr(op);
  debugAssignLayout(result, outputLayoutAttr);

  // Add secret.kernel attribute for MatmulDiagonal
  auto kernelAttr =
      secret::KernelAttr::get(ctx, KernelName::VecmatDiagonal, /*force=*/false);
  op->setAttr(secret::SecretDialect::kKernelAttrName, kernelAttr);

  return success();
}

LogicalResult LayoutPropagation::visitOperation(MatvecOp op) {
  auto matvecOp = cast<linalg::ContractionOpInterface>(*op);
  auto matrix = matvecOp.lhs();
  auto matrixType = cast<RankedTensorType>(matrix.getType());

  // Number of rows must be less than or equal to the number of columns.
  if (matrixType.getDimSize(0) > matrixType.getDimSize(1)) {
    return op->emitError() << "Matrix rows must be less than columns";
  }

  // TODO(#1597): a layout optimizer should really be selecting the diagonal
  // layout instead of this pass.

  LayoutAttr matrixLayout = assignedLayouts.at(matrix);
  // The Halevi-Shoup kernel (all we support at this time) requires one
  // ciphertext per matrix row.
  if (!isRelationSquatDiagonal(matrixType, ciphertextSize,
                               matrixLayout.getIntegerRelation())) {
    // Insert a layout conversion op to make the matrix layout squat diagonal
    MLIRContext* ctx = &getContext();
    mlir::IRRewriter builder(ctx);
    auto [toReplace, newMatrixLayoutAttr] =
        convertToLayout(ctx, builder, op, matrix, matrixLayout,
                        getDiagonalLayoutRelation(matrixType, ciphertextSize));
    debugAssignLayout(toReplace, newMatrixLayoutAttr);
    assignedLayouts.insert({toReplace, newMatrixLayoutAttr});
    matrix = toReplace;
  }

  // We also require the input to be row-major.
  auto vector = matvecOp.rhs();
  auto vectorType = cast<RankedTensorType>(vector.getType());
  LayoutAttr vectorLayout = assignedLayouts.at(vector);
  if (!isRelationRowMajor(vectorType, ciphertextSize,
                          vectorLayout.getIntegerRelation())) {
    // Insert a layout conversion op to make the matrix layout squat diagonal
    MLIRContext* ctx = &getContext();
    mlir::IRRewriter builder(ctx);
    auto [toReplace, newVectorLayoutAttr] =
        convertToLayout(ctx, builder, op, vector, vectorLayout,
                        getRowMajorLayoutRelation(vectorType, ciphertextSize));
    debugAssignLayout(toReplace, newVectorLayoutAttr);
    assignedLayouts.insert({toReplace, newVectorLayoutAttr});
    vector = toReplace;
  }

  // Always one result, and for the kernels we have right now it's always a
  // row-major replicated vector. Since the matrix may be rectangular, the
  // output layout may have different alignment from the input layout.
  auto result = matvecOp->getResult(0);
  RankedTensorType outputType = cast<RankedTensorType>(result.getType());
  FailureOr<LayoutAttr> outputLayoutResult = defaultLayoutForType(outputType);
  if (failed(outputLayoutResult)) {
    return failure();
  }
  LayoutAttr resultLayout = outputLayoutResult.value();

  assignedLayouts.insert({result, resultLayout});
  setResultLayoutAttr(op);
  debugAssignLayout(result, resultLayout);

  // Add secret.kernel attribute for MatvecDiagonal
  MLIRContext* ctx = &getContext();
  auto kernelAttr =
      secret::KernelAttr::get(ctx, KernelName::MatvecDiagonal, /*force=*/false);
  op->setAttr(secret::SecretDialect::kKernelAttrName, kernelAttr);

  return success();
}

LogicalResult LayoutPropagation::visitOperation(Conv2DOp op) {
  Value data = op.getInputs().front();
  Value filter = op.getInputs().back();
  auto dataType = cast<RankedTensorType>(data.getType());
  auto filterType = cast<RankedTensorType>(filter.getType());

  // Flattened data must fit into the ciphertext size.
  if (dataType.getNumElements() > ciphertextSize) {
    return op->emitOpError()
           << "Flattened data must fit into a single ciphertext, but got "
           << dataType.getNumElements() << " elements and ciphertext size is "
           << ciphertextSize;
  }

  MLIRContext* ctx = &getContext();
  mlir::IRRewriter builder(ctx);

  // TODO(#1597): a layout optimizer should really be selecting the
  // layout instead of this pass.
  LayoutAttr dataLayout = assignedLayouts.at(data);
  if (!isRelationRowMajor(dataType, ciphertextSize,
                          dataLayout.getIntegerRelation())) {
    auto [toReplace, newDataLayoutAttr] =
        convertToLayout(ctx, builder, op, data, dataLayout,
                        getRowMajorLayoutRelation(dataType, ciphertextSize));
    debugAssignLayout(toReplace, newDataLayoutAttr);
    assignedLayouts.insert({toReplace, newDataLayoutAttr});
  }

  // The kernel for this operation requires expanding the conv filter matrix
  // into a larger matrix and then diagonalizing.
  LayoutAttr filterLayout = assignedLayouts.at(filter);
  if (!isRelation2dConvFilterDiagonalized(filterType, dataType, /*padding=*/0,
                                          ciphertextSize,
                                          filterLayout.getIntegerRelation())) {
    // Insert a layout conversion op to make the matrix layout expanded and
    // squat diagonal
    auto convRelation = get2dConvFilterDiagonalizedRelation(
        filterType, dataType, /*padding=*/0, ciphertextSize);
    if (failed(convRelation)) {
      return failure();
    }
    auto [toReplace, newFilterLayoutAttr] = convertToLayout(
        ctx, builder, op, filter, filterLayout, convRelation.value());
    debugAssignLayout(toReplace, newFilterLayoutAttr);
    assignedLayouts.insert({toReplace, newFilterLayoutAttr});
  }

  // Always one result, and for the kernels we have right now it's always a
  // row-major replicated vector. Since the
  // output matrix will have different shape than the input, assign the new
  // layout.
  auto result = op->getResult(0);
  RankedTensorType outputType = cast<RankedTensorType>(result.getType());
  FailureOr<LayoutAttr> outputLayoutResult = defaultLayoutForType(outputType);
  if (failed(outputLayoutResult)) {
    return failure();
  }
  LayoutAttr resultLayout = outputLayoutResult.value();

  assignedLayouts.insert({result, resultLayout});
  setResultLayoutAttr(op);
  debugAssignLayout(result, resultLayout);

  // Add secret.kernel attribute for Conv2dMatvec
  auto kernelAttr =
      secret::KernelAttr::get(ctx, KernelName::MatvecDiagonal, /*force=*/false);
  op->setAttr(secret::SecretDialect::kKernelAttrName, kernelAttr);

  return success();
}

LogicalResult LayoutPropagation::visitOperation(MatmulOp op) {
  auto matmulOp = cast<linalg::ContractionOpInterface>(*op);
  auto inputMatrix = matmulOp.lhs();
  auto filterMatrix = matmulOp.rhs();
  if (!isSecret(inputMatrix, solver) || isSecret(filterMatrix, solver)) {
    // TODO(#1376): Implement bicyclic ciphertext-ciphertext matmul kernel.
    return failure();
  }

  MLIRContext* ctx = &getContext();
  mlir::IRRewriter builder(ctx);

  // Assign a per-row layout to the input matrix. Each row of the input matrix
  // will be packed into a separate ciphertext.
  auto inputMatrixType = cast<RankedTensorType>(inputMatrix.getType());
  LayoutAttr inputMatrixLayout = assignedLayouts.at(inputMatrix);
  if (!isRelationPerRow(inputMatrixType, ciphertextSize,
                        inputMatrixLayout.getIntegerRelation())) {
    // Insert a layout conversion op to make the matrix layout per-row
    auto [toReplace, newInputMatrixLayoutAttr] = convertToLayout(
        ctx, builder, op, inputMatrix, inputMatrixLayout,
        getPerRowLayoutRelation(inputMatrixType, ciphertextSize));
    debugAssignLayout(toReplace, newInputMatrixLayoutAttr);
    assignedLayouts.insert({toReplace, newInputMatrixLayoutAttr});
  }

  // Assign or change the filter matrix a diagonal layout.
  auto filterMatrixType = cast<RankedTensorType>(filterMatrix.getType());
  auto filterMatrixTransposeType = RankedTensorType::get(
      {filterMatrixType.getDimSize(1), filterMatrixType.getDimSize(0)},
      filterMatrixType.getElementType());
  LayoutAttr filterMatrixLayout = assignedLayouts.at(filterMatrix);
  IntegerRelation filterMatrixLayoutRelation =
      filterMatrixLayout.getIntegerRelation();
  auto clonedFilterMatrixRelation = filterMatrixLayoutRelation.clone();
  auto domainOffset =
      filterMatrixLayoutRelation.getVarKindOffset(presburger::VarKind::Domain);
  clonedFilterMatrixRelation->swapVar(domainOffset, domainOffset + 1);
  if (!isRelationSquatDiagonal(filterMatrixTransposeType, ciphertextSize,
                               *clonedFilterMatrixRelation)) {
    // Insert a layout conversion op to make the matrix layout squat diagonal
    auto [toReplace, newFilterMatrixLayoutAttr] = convertToLayout(
        ctx, builder, op, filterMatrix, filterMatrixLayout,
        getDiagonalLayoutRelation(filterMatrixTransposeType, ciphertextSize));
    debugAssignLayout(toReplace, newFilterMatrixLayoutAttr);
    assignedLayouts.insert({toReplace, newFilterMatrixLayoutAttr});
  }

  // The output has the same per-row layout as the input matrix.
  auto result = matmulOp->getResult(0);
  RankedTensorType outputType = cast<RankedTensorType>(result.getType());
  IntegerRelation outputLayoutResult =
      getPerRowLayoutRelation(outputType, ciphertextSize);
  LayoutAttr outputLayoutAttr =
      LayoutAttr::getFromIntegerRelation(ctx, outputLayoutResult);

  assignedLayouts.insert({result, outputLayoutAttr});
  setResultLayoutAttr(op);
  debugAssignLayout(result, outputLayoutAttr);

  // Add secret.kernel attribute for MatmulDiagonal
  auto kernelAttr =
      secret::KernelAttr::get(ctx, KernelName::MatmulDiagonal, /*force=*/false);
  op->setAttr(secret::SecretDialect::kKernelAttrName, kernelAttr);

  return success();
}

LogicalResult LayoutPropagation::visitOperation(ReduceOp op) {
  for (const auto& [tensor, result] :
       llvm::zip(op.getInputs(), op.getResults())) {
    LayoutAttr resultLayout =
        convertLayoutForReduce(assignedLayouts.at(tensor), op.getDimensions());
    assignedLayouts.insert({result, resultLayout});
    debugAssignLayout(result, resultLayout);
  }
  setResultLayoutAttr(op);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(affine::AffineForOp op) {
  // Transfer the layout of the inits to the region iter args
  for (const auto& [init, iterArg, result] :
       llvm::zip(op.getInits(), op.getRegionIterArgs(), op.getResults())) {
    LayoutAttr layout;

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
      layout = cast<LayoutAttr>(res.value().getLayout());
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

LogicalResult LayoutPropagation::visitOperation(tensor::ExtractOp op) {
  std::string relationStr = "{ [] -> [ct, slot] : ct = 0 and slot = 0 }";
  LayoutAttr scalarLayout = LayoutAttr::get(op.getContext(), relationStr);
  LLVM_DEBUG(llvm::dbgs() << "Assigning scalar layout " << scalarLayout
                          << " to value " << op.getResult() << "\n");
  Value result = op.getResult();
  assignedLayouts.insert({result, scalarLayout});
  debugAssignLayout(result, scalarLayout);
  setResultLayoutAttr(op);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(tensor::InsertOp op) {
  // The layout of the result is the same as the layout of the input tensor.
  // The scalar input is handled by the generic logic in visitOperation,
  // which will assign a default scalar layout if it is secret and has no
  // layout.
  LayoutAttr tensorLayout = assignedLayouts.at(op.getDest());
  Value result = op.getResult();
  assignedLayouts.insert({result, tensorLayout});
  debugAssignLayout(result, tensorLayout);
  setResultLayoutAttr(op);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(tensor::InsertSliceOp op) {
  // Set the result layout to be the same as the dest tensor.
  if (!assignedLayouts.contains(op.getDest())) {
    return op->emitError() << "Destination tensor has no assigned layout";
  }
  LayoutAttr destLayout = assignedLayouts.at(op.getDest());
  Value result = op.getResult();
  assignedLayouts.insert({result, destLayout});
  debugAssignLayout(result, destLayout);
  setResultLayoutAttr(op);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(tensor::ExtractSliceOp op) {
  // Assign the induced layout from extracting a slice from the source tensor.
  if (!assignedLayouts.contains(op.getSource())) {
    return op->emitError() << "Source tensor has no assigned layout";
  }
  IntegerRelation sourceLayout =
      assignedLayouts.at(op.getSource()).getIntegerRelation();

  FailureOr<IntegerRelation> maybeSliceExtractionLayout =
      getSliceExtractionRelation(op.getSourceType(), op.getResultType(),
                                 SmallVector<int64_t>(op.getStaticOffsets()),
                                 SmallVector<int64_t>(op.getStaticSizes()),
                                 SmallVector<int64_t>(op.getStaticStrides()));
  if (failed(maybeSliceExtractionLayout)) {
    return failure();
  }
  IntegerRelation sliceExtractionLayout = maybeSliceExtractionLayout.value();

  // Compose the inverted slice extraction layout with the source layout to
  // get the result slice layout.
  sliceExtractionLayout.inverse();
  sliceExtractionLayout.compose(sourceLayout);
  // If the slice extracted was not at offset zero, then the resulting slice may
  // be indexed at a non-zero ciphertext. For example, imagine extracting a
  // slice out of the second ciphertext. Then computing the inverse of the slice
  // extraction layout and composing that with the source relation would mean
  // that the slice would map to the second ciphertext. But a slice extracted
  // from a tensor.extract_slice op is always indexed starting from zero.
  // Reindexing the the resulting relation to start from ciphertext zero.
  auto ctVarOffset =
      sliceExtractionLayout.getVarKindOffset(presburger::VarKind::Range);
  auto ctLowerBound = sliceExtractionLayout.getConstantBound64(
      presburger::BoundType::LB, ctVarOffset);
  if (!ctLowerBound) {
    return op.emitError() << "failed to get constant bound on ciphertext index";
  }
  auto zeroIndexedSliceLayout =
      shiftVar(sliceExtractionLayout, ctVarOffset, -ctLowerBound.value());

  LayoutAttr outputLayout = LayoutAttr::getFromIntegerRelation(
      op.getContext(), zeroIndexedSliceLayout);
  assignedLayouts.insert({op.getResult(), outputLayout});
  debugAssignLayout(op.getResult(), outputLayout);
  setResultLayoutAttr(op);
  return success();
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
    Operation* op) {
  return TypeSwitch<Operation*, CompatibilityResult>(op)
      // Trivially true ops
      .Case<func::FuncOp, GenericOp, YieldOp, affine::AffineForOp,
            affine::AffineYieldOp>(
          [&](auto op) { return CompatibilityResult{true, std::nullopt}; })
      // Ops with special rules
      .Case<ReduceOp, MatvecOp, VecmatOp, Conv2DOp, tensor::InsertSliceOp>(
          [&](auto op) { return hasCompatibleArgumentLayouts(op); })
      // By default, assume operands must all have the same layout.
      .Default([&](Operation* op) {
        std::optional<LayoutAttr> firstFoundLayout;
        // All operands require a layout unless the op implements
        // OperandLayoutRequirementOpInterface, which allows it to specify which
        // subset of operands require layouts.
        auto layoutReqIface = dyn_cast<OperandLayoutRequirementOpInterface>(op);

        for (auto& opOperand : op->getOpOperands()) {
          bool secretness = isSecret(opOperand.get(), solver);
          if (layoutReqIface && !layoutReqIface.operandRequiresLayout(
                                    opOperand.getOperandNumber(), secretness)) {
            continue;
          }

          if (!assignedLayouts.contains(opOperand.get())) {
            // If the operand has no layout, we can't propagate layout
            // information to the result.
            return CompatibilityResult{
                false, op->emitError("operand has no assigned layout")};
          }
          LayoutAttr layout = assignedLayouts.at(opOperand.get());

          if (!firstFoundLayout.has_value()) firstFoundLayout = layout;
          if (layout != firstFoundLayout.value()) {
            return CompatibilityResult{false, std::nullopt};
          }
        }

        return CompatibilityResult{true, std::nullopt};
      });
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
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

    LayoutAttr inputLayout = assignedLayouts.at(input);
    LayoutAttr initLayout = assignedLayouts.at(init);
    LayoutAttr reducedInputLayout =
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

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
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

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
    Conv2DOp op) {
  // Currently only support secret data and plaintext filters.
  Value data = op.getInputs().front();
  Value filter = op.getInputs().back();
  if (isSecret(filter, solver) || !isSecret(data, solver)) {
    return {false, op->emitError("Only secret data and plaintext filters are "
                                 "supported for linalg.conv2d")};
  }

  if (!assignedLayouts.contains(data)) {
    return {false, op->emitError("data operand has no assigned layout")};
  }
  return {true, std::nullopt};
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
    tensor::InsertSliceOp op) {
  // The arguments of a tensor::InsertSliceOp are the tensors to insert and the
  // tensor to insert into.
  auto insert = op.getSource();
  auto dest = op.getDest();

  if (!assignedLayouts.contains(insert)) {
    return {false, op->emitError("input tensor has no assigned layout")};
  }
  if (!assignedLayouts.contains(dest)) {
    return {false, op->emitError("destination tensor has no assigned layout")};
  }

  auto insertLayout = assignedLayouts.at(insert);
  auto destLayout = assignedLayouts.at(dest);
  auto destRelation = destLayout.getIntegerRelation();
  auto compatibleDestRelation = pushSliceLayoutThroughInsertSlice(
      SmallVector<int64_t>(op.getStaticSizes()), op.getResultType().getShape(),
      insertLayout.getIntegerRelation());
  assert(succeeded(compatibleDestRelation) && "expected to infer slice layout");

  // We compare layout attributes, which is a strict equality check on the
  // attribute string. This is more strict than comparing the IntegerRelations,
  // but equality checking on the relations is expensive.
  if (LayoutAttr::getFromIntegerRelation(
          op.getContext(), compatibleDestRelation.value()) != destLayout) {
    return {false, std::nullopt};
  }

  return {true, std::nullopt};
}

void LayoutPropagation::rectifyIncompatibleOperandLayouts(Operation* op) {
  LLVM_DEBUG({
    auto diag = op->emitRemark() << "Inserting layout conversion op due to "
                                    "disagreeing operand layouts";
    auto& note = diag.attachNote();
    for (auto operand : op->getOperands()) {
      LayoutAttr operandLayout;
      if (assignedLayouts.contains(operand))
        operandLayout = assignedLayouts.at(operand);
      note << "\n- Operand: " << operand << "; Layout: " << operandLayout;
    }
  });

  TypeSwitch<Operation*>(op)
      // Ops with special rules
      .Case<ReduceOp, tensor::InsertOp, tensor::InsertSliceOp>(
          [&](auto op) { return rectifyIncompatibleOperandLayouts(op); })
      .Default([&](Operation* op) {
        // Default target layout is chosen arbitrarily as the first operand's
        // layout for now. A different pass is responsible for optimizing the
        // placement and mechanics of the layout conversion ops.
        mlir::IRRewriter builder(&getContext());
        const auto it = llvm::find_if(op->getOperands(), [this](Value pair) {
          return assignedLayouts.contains(pair);
        });
        LayoutAttr targetLayout = assignedLayouts.at(*it);

        for (auto& opOperand : op->getOpOperands()) {
          if (!assignedLayouts.contains(opOperand.get())) continue;
          LayoutAttr sourceLayout = assignedLayouts.at(opOperand.get());

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

void LayoutPropagation::rectifyIncompatibleOperandLayouts(ReduceOp op) {
  mlir::IRRewriter builder(&getContext());
  builder.setInsertionPoint(op);

  for (const auto& [input, init] : llvm::zip(op.getInputs(), op.getInits())) {
    LayoutAttr inputLayout = assignedLayouts.at(input);
    LayoutAttr initLayout = assignedLayouts.at(init);
    LayoutAttr reducedInputLayout =
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

void LayoutPropagation::rectifyIncompatibleOperandLayouts(
    tensor::InsertSliceOp op) {
  // Update the dest tensor to align with the source tensor slice.
  LayoutAttr sliceLayout = assignedLayouts.at(op.getSource());
  auto maybeNewLayout = pushSliceLayoutThroughInsertSlice(
      SmallVector<int64_t>(op.getStaticSizes()), op.getResultType().getShape(),
      sliceLayout.getIntegerRelation());
  assert(succeeded(maybeNewLayout) && "expected to infer slice layout");
  auto newLayoutAttr = LayoutAttr::getFromIntegerRelation(
      op.getContext(), maybeNewLayout.value());

  if (!assignedLayouts.contains(op.getDest())) {
    assignedLayouts.insert({op.getDest(), newLayoutAttr});
    return;
  }

  mlir::IRRewriter builder(&getContext());
  builder.setInsertionPoint(op);
  LayoutAttr destLayout = assignedLayouts.at(op.getDest());
  auto convertLayoutOp = ConvertLayoutOp::create(
      builder, op->getLoc(), op.getDest(), destLayout, newLayoutAttr);
  Value toReplace = convertLayoutOp.getResult();
  builder.replaceUsesWithIf(op.getDest(), toReplace, [&](OpOperand& operand) {
    return operand.getOwner() == op;
  });
  assignedLayouts.insert({toReplace, newLayoutAttr});
  setResultLayoutAttr(convertLayoutOp);
}

void LayoutPropagation::rectifyIncompatibleOperandLayouts(tensor::InsertOp op) {
  // The scalar input's layout must be made to match the tensor input's layout.
  // Ideally we could be smarter here, for example to take a scalar layout
  // which is a single slot in a single ciphertext, and insert it into the
  // correct slot of the single ciphertext of the dest tensor via a mask and
  // rotate. But this kernel is not expected to be used with ciphertext inputs,
  // so we instead incur the cost of a layout conversion before the insert.
  mlir::IRRewriter builder(&getContext());
  builder.setInsertionPoint(op);

  LayoutAttr destLayout = assignedLayouts.at(op.getDest());
  LayoutAttr scalarLayout = assignedLayouts.at(op.getScalar());
  IntegerRelation destRel = destLayout.getIntegerRelation();
  std::optional<int64_t> destNumCts = destRel.getConstantBound64(
      presburger::BoundType::UB,
      // First range var is the ciphertext index, always has lower bound of
      // zero
      destRel.getVarKindOffset(presburger::VarKind::Range));
  assert(destNumCts.has_value());

  std::string newScalarLayoutStr = llvm::formatv(
      "{ [] -> [ct, slot] : 0 <= ct <= {0} and 0 <= slot <= {1} }",
      destNumCts.value(), ciphertextSize - 1);
  LayoutAttr newScalarLayout =
      LayoutAttr::get(op.getContext(), newScalarLayoutStr);

  ConvertLayoutOp convertOp = ConvertLayoutOp::create(
      builder, op->getLoc(), op.getScalar(), scalarLayout, newScalarLayout);
  Value toReplace = convertOp.getResult();
  builder.replaceUsesWithIf(op.getScalar(), toReplace, [&](OpOperand& operand) {
    return operand.getOwner() == op;
  });
  assignedLayouts.insert({toReplace, newScalarLayout});
  setResultLayoutAttr(convertOp);
}

void LayoutPropagation::passLayoutThroughOp(Operation* op) {
  // All inputs have the same layout, so just propagate it to all results
  LayoutAttr layout = assignedLayouts.at(op->getOperand(0));
  for (Value result : op->getResults()) {
    assignedLayouts.insert({result, layout});
    debugAssignLayout(result, layout);
  }
  setResultLayoutAttr(op);
}

FailureOr<LayoutAttr> LayoutPropagation::defaultLayoutForScalarType(
    Type scalarType) {
  auto* ctx = scalarType.getContext();
  OpBuilder builder(ctx);

  // Only support Int/Index/Float scalars,
  // fail for any other random type that might make it here.
  if (!isa<FloatType, IntegerType, IndexType>(scalarType)) {
    return failure();
  }

  std::string relationStr = llvm::formatv(
      "{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= {0} }", ciphertextSize - 1);
  return LayoutAttr::get(ctx, relationStr);
}

FailureOr<LayoutAttr> LayoutPropagation::defaultLayoutForType(Type type) {
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
  return LayoutAttr::getFromIntegerRelation(tensorType.getContext(), relation);
}

void LayoutPropagation::setResultLayoutAttr(Operation* op) {
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

void LayoutPropagation::runOnOperation() {
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

  RewritePatternSet cleanupPatterns(&getContext());
  cleanupPatterns.add<tensor_ext::FoldConvertLayoutIntoAssignLayoutPattern>(
      &getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(cleanupPatterns));
};

}  // namespace heir
}  // namespace mlir
