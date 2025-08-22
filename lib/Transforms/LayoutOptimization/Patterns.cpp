#include "lib/Transforms/LayoutOptimization/Patterns.h"

#include <optional>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {

using tensor_ext::AssignLayoutOp;
using tensor_ext::ConvertLayoutOp;

namespace {
auto& kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;
}

LogicalResult tryFoldLayoutConversionIntoPrevious(
    RewriterBase& rewriter, ConvertLayoutOp op,
    SmallVector<Operation*>& opsToErase) {
  if (auto priorConversion = op.getValue().getDefiningOp<ConvertLayoutOp>()) {
    // merge the two conversions into one
    auto newFrom = priorConversion.getFromLayoutAttr();
    auto newTo = op.getToLayoutAttr();
    auto newConversion = ConvertLayoutOp::create(
        rewriter, op.getLoc(), priorConversion.getValue(), newFrom, newTo);
    newConversion->setAttr(kLayoutAttrName, newTo);

    rewriter.replaceAllUsesWith(op, newConversion);
    opsToErase.push_back(op);
    if (priorConversion->use_empty()) {
      opsToErase.push_back(priorConversion);
    }
    return success();
  }

  if (auto priorAssignment = op.getValue().getDefiningOp<AssignLayoutOp>()) {
    // merge the conversion into the assignment return success();
    auto newLayout = op.getToLayoutAttr();
    auto newAssign = AssignLayoutOp::create(
        rewriter, op.getLoc(), priorAssignment.getValue(), newLayout);
    newAssign->setAttr(kLayoutAttrName, newLayout);

    rewriter.replaceAllUsesWith(op, newAssign);
    opsToErase.push_back(op);
    if (priorAssignment->use_empty()) {
      opsToErase.push_back(priorAssignment);
    }
    return success();
  }

  return failure();
}

LogicalResult FoldLayoutConversions::matchAndRewrite(
    tensor_ext::ConvertLayoutOp op, PatternRewriter& rewriter) const {
  SmallVector<Operation*> opsToErase;
  LogicalResult result =
      tryFoldLayoutConversionIntoPrevious(rewriter, op, opsToErase);

  for (auto* op : opsToErase) {
    rewriter.eraseOp(op);
  }

  return result;
}

LogicalResult HoistArgLayouts::matchAndRewrite(
    func::FuncOp func, PatternRewriter& rewriter) const {
  bool changed = false;

  auto getFirstLayoutConversionOp =
      [&](OpOperand& use) -> std::optional<ConvertLayoutOp> {
    auto genericOp = dyn_cast<secret::GenericOp>(use.getOwner());
    if (!genericOp) {
      // A block argument may be returned directly, so no layout conversion is
      // performed.
      return std::nullopt;
    }
    auto innerArg = genericOp.getBody(0)->getArgument(use.getOperandNumber());
    if (!innerArg.hasOneUse()) return std::nullopt;
    // Any further layout conversions after operations that don't change layout
    // were already hoisted through in the layout-optimization pass.
    auto convertLayoutOp =
        dyn_cast<ConvertLayoutOp>(*innerArg.getUsers().begin());
    if (!convertLayoutOp) return std::nullopt;

    return convertLayoutOp;
  };

  for (auto& blockArg : func.getArguments()) {
    // Check that all uses of the block argument have the same layout
    // conversion. Otherwise, hoisting may not produce a benefit; it would
    // require duplicating the function argument or updating the layout
    // conversions of other uses.
    auto maybeLayoutOps =
        llvm::map_range(blockArg.getUses(), getFirstLayoutConversionOp);
    if (maybeLayoutOps.empty()) continue;
    auto maybeLayouts = llvm::map_range(
        llvm::to_vector(maybeLayoutOps),
        [](auto& maybeLayoutOp) -> std::optional<Attribute> {
          if (!maybeLayoutOp.has_value()) return std::nullopt;
          // TODO(#2047): Hoisting new layouts is unsupported.
          auto toLayout = maybeLayoutOp.value().getToLayoutAttr();
          if (!toLayout) return std::nullopt;
          return toLayout;
        });
    if (!llvm::all_equal(maybeLayouts)) continue;

    // If there was no layout conversion, there is nothing to hoist.
    auto maybeLayoutOp = *maybeLayoutOps.begin();
    if (!maybeLayoutOp.has_value()) continue;

    Attribute toLayout = maybeLayoutOp.value().getToLayoutAttr();

    rewriter.modifyOpInPlace(func, [&]() {
      // Update the function argument layout and the layout conversion's input
      // attribute
      func.setArgAttr(blockArg.getArgNumber(), kLayoutAttrName, toLayout);
      for (auto maybeLayoutOp : maybeLayoutOps) {
        auto innerValue = maybeLayoutOp.value().getValue();
        setAttributeAssociatedWith(innerValue, kLayoutAttrName, toLayout);
      }
    });
    for (auto maybeLayoutOp : maybeLayoutOps) {
      rewriter.replaceOp(maybeLayoutOp.value(),
                         {maybeLayoutOp.value().getValue()});
    }
    changed = true;
  }
  return changed ? success() : failure();
}

}  // namespace heir
}  // namespace mlir
