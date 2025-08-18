
#include "lib/Transforms/LayoutOptimization/Patterns.h"

#include <optional>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
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

namespace {}  // namespace

LogicalResult HoistArgLayouts::matchAndRewrite(
    func::FuncOp func, PatternRewriter& rewriter) const {
  bool changed = false;

  auto getFirstLayoutConversionOp =
      [&](OpOperand& use) -> std::optional<tensor_ext::ConvertLayoutOp> {
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
        dyn_cast<tensor_ext::ConvertLayoutOp>(*innerArg.getUsers().begin());
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
        [](auto& maybeLayoutOp) -> std::optional<tensor_ext::LayoutAttr> {
          if (!maybeLayoutOp.has_value()) return std::nullopt;
          // TODO(#2047): Hoisting new layouts is unsupported.
          auto toLayout = dyn_cast<tensor_ext::LayoutAttr>(
              maybeLayoutOp.value().getToLayoutAttr());
          if (!toLayout) return std::nullopt;
          return toLayout;
        });
    if (!llvm::all_equal(maybeLayouts)) continue;

    // If there was no layout conversion, there is nothing to hoist.
    auto maybeLayoutOp = *maybeLayoutOps.begin();
    if (!maybeLayoutOp.has_value()) continue;

    tensor_ext::LayoutAttr toLayout =
        cast<tensor_ext::LayoutAttr>(maybeLayoutOp.value().getToLayoutAttr());

    rewriter.modifyOpInPlace(func, [&]() {
      // Update the function argument layout and the layout conversion's input
      // attribute
      func.setArgAttr(blockArg.getArgNumber(),
                      tensor_ext::TensorExtDialect::kLayoutAttrName, toLayout);
      for (auto maybeLayoutOp : maybeLayoutOps) {
        auto innerValue = maybeLayoutOp.value().getValue();
        setAttributeAssociatedWith(
            innerValue, tensor_ext::TensorExtDialect::kLayoutAttrName,
            toLayout);
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
