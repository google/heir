#include "lib/Transforms/LayoutOptimization/LayoutOptimization.h"

#include <cassert>
#include <cstdint>
#include <tuple>
#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/LayoutOptimization/Patterns.h"
#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Iterators.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "layout-optimization"

namespace mlir {
namespace heir {

using ::mlir::arith::AddFOp;
using ::mlir::arith::AddIOp;
using ::mlir::heir::tensor_ext::AssignLayoutOp;
using ::mlir::heir::tensor_ext::ConvertLayoutOp;
using ::mlir::linalg::ReduceOp;
using tensor_ext::LayoutAttr;

#define GEN_PASS_DEF_LAYOUTOPTIMIZATION
#include "lib/Transforms/LayoutOptimization/LayoutOptimization.h.inc"

namespace {

auto &kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;

typedef int64_t Cost;

// TODO(#1595): Implement a more accurate cost model.
Cost computeCostOfLayoutConversion(LayoutAttr fromLayout, LayoutAttr toLayout) {
  return (fromLayout == toLayout) ? 0 : 1;
}

struct OperandChange {
  LayoutAttr fromLayout;
  LayoutAttr toLayout;
  Cost cost;
};

struct HoistingResult {
  // Net increase in cost.
  Cost cost;
  SmallVector<LayoutAttr> newInputLayouts;
  LayoutAttr newOutputLayout;
  ConvertLayoutOp convertLayoutOp;
};

}  // namespace

struct LayoutOptimization : impl::LayoutOptimizationBase<LayoutOptimization> {
  using LayoutOptimizationBase::LayoutOptimizationBase;

  enum OpHoistResult { UNHOISTABLE, SUCCESS, FAILURE };
  OpHoistResult hoistOp(Operation *op, IRRewriter &builder);

  HoistingResult computeHoistedCost(Operation *kernel,
                                    ConvertLayoutOp convertLayoutOp);

  // Computes cost of changed operand.
  OperandChange costOfChangedOperand(OpOperand &operand, Operation *kernel,
                                     LayoutAttr newLayout);

  // Computes cost of changed result.
  Cost costOfChangedResult(Operation *kernel, LayoutAttr newLayout);

  void runOnOperation() override;
};

void LayoutOptimization::runOnOperation() {
  auto ctx = &getContext();
  IRRewriter builder(ctx);
  WalkResult result =
      getOperation()->walk<WalkOrder::PreOrder, ReverseIterator>(
          [&](Operation *op) {
            // Attempt to hoist layout conversions before this operation.
            OpHoistResult result = hoistOp(op, builder);
            if (result == FAILURE) {
              return WalkResult::interrupt();
            };
            // TODO(#1598): Fold newly added convert_layout ops after each
            // rewrite.
            return WalkResult::advance();
          });

  if (result.wasInterrupted()) {
    signalPassFailure();
  }

  RewritePatternSet patterns(ctx);
  patterns.add<HoistArgLayouts>(ctx);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
};

LayoutOptimization::OpHoistResult LayoutOptimization::hoistOp(
    Operation *op, IRRewriter &builder) {
  if (isa<ConvertLayoutOp>(op)) {
    // Folders will canonicalize any hoistable convert layout ops.
    return UNHOISTABLE;
  }

  if (op->getNumResults() != 1) {
    // Most ops we're interested in have a single result.
    return UNHOISTABLE;
  }

  LLVM_DEBUG(llvm::dbgs() << "Considering hoisting op: " << op->getName()
                          << " with layout "
                          << findAttributeAssociatedWith(op->getResult(0),
                                                         kLayoutAttrName)
                          << "\n");

  // Check if any results are converted directly after.
  SmallVector<ConvertLayoutOp> resultLayoutConversions;
  for (auto user : op->getResult(0).getUsers()) {
    if (auto convertLayoutOp = dyn_cast<ConvertLayoutOp>(user)) {
      resultLayoutConversions.push_back(convertLayoutOp);
    }
  }

  // No result has a layout conversion directly after.
  if (resultLayoutConversions.empty()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Skipping op, no results needed immediate layout conversion\n");
    return UNHOISTABLE;
  }

  // Now compute the cost of hoisting each conversion layout.
  SmallVector<HoistingResult> hoistingResults;
  for (auto resultLayoutConversion : resultLayoutConversions) {
    auto result = computeHoistedCost(op, resultLayoutConversion);
    LLVM_DEBUG(llvm::dbgs() << "\tHoisting layout " << result.newOutputLayout
                            << " will cost " << result.cost << "\n");
    hoistingResults.push_back(result);
  }

  // Select the least costly layout conversion to hoist.
  auto minHoistingResult = llvm::min_element(
      hoistingResults, [](const HoistingResult &a, const HoistingResult &b) {
        return a.cost < b.cost;
      });

  if (minHoistingResult->cost > 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping op, all hoisting results have positive net cost\n");
    return UNHOISTABLE;
  }

  LLVM_DEBUG(llvm::dbgs()
             << "Minimum cost layout conversion converts output layout from "
             << op->getAttr(kLayoutAttrName) << " to layout "
             << minHoistingResult->newOutputLayout << "\n");

  builder.setInsertionPoint(op);
  for (auto i = 0; i < op->getNumOperands(); ++i) {
    // Convert each operand layout given the result.
    auto operand = op->getOperand(i);
    auto originalLayout = findAttributeAssociatedWith(operand, kLayoutAttrName);
    if (failed(originalLayout)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to get layout for operand : " << i << "\n");
      return FAILURE;
    }
    LLVM_DEBUG(llvm::dbgs() << "Converting operand " << i << " with layout "
                            << originalLayout.value() << " to layout "
                            << minHoistingResult->newInputLayouts[i] << "\n");
    LayoutAttr newInputLayout = minHoistingResult->newInputLayouts[i];
    auto newInput = builder.create<ConvertLayoutOp>(
        op->getLoc(), operand, cast<LayoutAttr>(originalLayout.value()),
        newInputLayout);
    newInput->setAttr(kLayoutAttrName, newInputLayout);
    builder.replaceUsesWithIf(operand, newInput, [&](OpOperand &operand) {
      return operand.getOwner() == op;
    });
  }

  // Set new layout attribute for the op.
  LayoutAttr newOutputLayout = minHoistingResult->newOutputLayout;
  op->setAttr(kLayoutAttrName, newOutputLayout);

  // Replace uses of the convert layout op with its input.
  builder.replaceAllUsesWith(minHoistingResult->convertLayoutOp.getResult(),
                             minHoistingResult->convertLayoutOp.getValue());

  // Ensure that any other convert_layout ops have their from_layouts updated.
  // Update downstream ops.
  for (auto *user : op->getResult(0).getUsers()) {
    if (auto convertLayoutOp = dyn_cast<ConvertLayoutOp>(user)) {
      // Update any convert_layout uses of the op result.
      convertLayoutOp.setFromLayoutAttr(newOutputLayout);
    }
  }

  builder.eraseOp(minHoistingResult->convertLayoutOp);
  return SUCCESS;
}

OperandChange LayoutOptimization::costOfChangedOperand(OpOperand &operand,
                                                       Operation *kernel,
                                                       LayoutAttr newLayout) {
  auto value = operand.get();
  if (dyn_cast_or_null<AssignLayoutOp>(value.getDefiningOp())) {
    // TODO(#1596): Use a proper analysis to determine whether a value's layout
    // is free to change, rather than relying on tensor_ext.assign_layout.
    return OperandChange{LayoutAttr(), LayoutAttr(), 0};
  }
  if (auto convertLayoutOp = value.getDefiningOp<ConvertLayoutOp>()) {
    // If the operand came from convert_layout, the cost of the change is
    // (folded conversion - original conversion).
    auto fromLayout = convertLayoutOp.getFromLayout();
    Cost originalConversion = computeCostOfLayoutConversion(
        fromLayout, convertLayoutOp.getToLayout());
    Cost foldedConversion =
        computeCostOfLayoutConversion(fromLayout, newLayout);
    return OperandChange{fromLayout, newLayout,
                         foldedConversion - originalConversion};
  }
  // Otherwise, we need to insert a new convert_layout op.
  auto originalLayoutResult =
      findAttributeAssociatedWith(value, kLayoutAttrName);
  assert(succeeded(originalLayoutResult) &&
         "Operand does not have a layout attribute");
  auto originalLayout = cast<LayoutAttr>(originalLayoutResult.value());
  return OperandChange{
      originalLayout, newLayout,
      computeCostOfLayoutConversion(originalLayout, newLayout)};
}

Cost LayoutOptimization::costOfChangedResult(Operation *kernel,
                                             LayoutAttr newLayout) {
  Cost totalCost = 0;
  for (auto user : kernel->getResult(0).getUsers()) {
    if (auto convertLayoutOp = dyn_cast<ConvertLayoutOp>(user)) {
      Cost originalConversion = computeCostOfLayoutConversion(
          convertLayoutOp.getFromLayout(), convertLayoutOp.getToLayout());
      Cost foldedConversion = computeCostOfLayoutConversion(
          newLayout, convertLayoutOp.getToLayout());
      totalCost += foldedConversion - originalConversion;
    }
  }
  return totalCost;
}

HoistingResult LayoutOptimization::computeHoistedCost(
    Operation *kernel, ConvertLayoutOp convertLayoutOp) {
  HoistingResult result =
      llvm::TypeSwitch<Operation *, HoistingResult>(kernel)
          .Case<ReduceOp>([&](auto kernel) -> HoistingResult {
            // TODO(#1597): Add linalg::reduce op kernel.
            return HoistingResult();
          })
          .Case<AddIOp, AddFOp>([&](Operation *kernel) -> HoistingResult {
            HoistingResult result;
            auto outputLayout = convertLayoutOp.getToLayout();
            result.cost = 0;
            // A map is used to deduplicate operand changes.
            DenseMap<std::tuple<Value, LayoutAttr, LayoutAttr>, Cost>
                operandChangeMap;
            for (auto &operand : kernel->getOpOperands()) {
              auto computedCost =
                  costOfChangedOperand(operand, kernel, outputLayout);
              operandChangeMap[std::make_tuple(
                  operand.get(), computedCost.fromLayout,
                  computedCost.toLayout)] = computedCost.cost;
            }
            for (auto &[_, cost] : operandChangeMap) {
              result.cost += cost;
            }
            result.cost += costOfChangedResult(kernel, outputLayout);
            SmallVector<LayoutAttr> newInputLayouts(kernel->getNumOperands(),
                                                    outputLayout);
            result.newOutputLayout = outputLayout;
            result.newInputLayouts = newInputLayouts;
            result.convertLayoutOp = convertLayoutOp;
            return result;
          })
          .Default(
              [](auto kernel) -> HoistingResult { return HoistingResult(); });
  return result;
}

}  // namespace heir
}  // namespace mlir
