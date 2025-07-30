#include "lib/Transforms/LayoutOptimization/LayoutOptimization.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/Kernel.h"
#include "lib/Transforms/LayoutOptimization/Hoisting.h"
#include "lib/Transforms/LayoutOptimization/LayoutConversionCost.h"
#include "lib/Transforms/LayoutOptimization/Patterns.h"
#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
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

using ::mlir::heir::secret::KernelAttr;
using ::mlir::heir::tensor_ext::AssignLayoutOp;
using ::mlir::heir::tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;

constexpr const static StringLiteral kKernelAttrName =
    ::mlir::heir::secret::SecretDialect::kKernelAttrName;

#define GEN_PASS_DEF_LAYOUTOPTIMIZATION
#include "lib/Transforms/LayoutOptimization/LayoutOptimization.h.inc"

namespace {

auto &kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;

using Cost = int64_t;

struct OperandChange {
  LayoutAttr fromLayout;
  LayoutAttr toLayout;
  Cost cost;
};

struct HoistOption {
  // Net increase in cost.
  Cost cost;
  HoistResult hoistResult;
};

}  // namespace

struct LayoutOptimization : impl::LayoutOptimizationBase<LayoutOptimization> {
  using LayoutOptimizationBase::LayoutOptimizationBase;

  enum OpHoistResult { UNHOISTABLE, SUCCESS, FAILURE };
  OpHoistResult hoistOp(Operation *op, IRRewriter &builder);

  std::vector<HoistOption> computeHoistingOptions(
      Operation *op, ConvertLayoutOp convertLayoutOp);

  // Computes cost of changed operand.
  OperandChange costOfChangedOperand(OpOperand &operand, Operation *kernel,
                                     LayoutAttr newLayout);

  // Computes cost of changed result.
  Cost costOfChangedResult(Operation *kernel, LayoutAttr newLayout);

  void runOnOperation() override;
};

void LayoutOptimization::runOnOperation() {
  auto *ctx = &getContext();
  IRRewriter builder(ctx);
  WalkResult result =
      getOperation()->walk<WalkOrder::PreOrder, ReverseIterator>(
          [&](Operation *op) {
            if (auto hoistable =
                    dyn_cast<LayoutConversionHoistableOpInterface>(op)) {
              // TODO(#1888): figure out how to get OpInterface verifier to run
              // automatically.
              KernelName kernelName = KernelName::Trivial;
              auto attrName =
                  ::mlir::heir::secret::SecretDialect::kKernelAttrName;
              if (op->hasAttr(attrName)) {
                kernelName =
                    op->getAttrOfType<::mlir::heir::secret::KernelAttr>(
                          attrName)
                        .getName();
              }

              if (!::mlir::heir::isSupportedKernel(op, kernelName)) {
                op->emitOpError() << "has unsupported kernel\n";
                return WalkResult::interrupt();
              }
            }

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
  for (auto *user : op->getResult(0).getUsers()) {
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
  SmallVector<HoistOption> hoistingOptions;
  for (auto resultLayoutConversion : resultLayoutConversions) {
    auto options = computeHoistingOptions(op, resultLayoutConversion);
    for (auto &option : options) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tHoisting layout " << option.hoistResult.newOutputLayout
                 << " via kernel " << option.hoistResult.newKernel
                 << " will cost " << option.cost << "\n");

      if (option.hoistResult.newInputLayouts.size() != op->getNumOperands()) {
        LLVM_DEBUG({
          std::string kernelName;
          llvm::raw_string_ostream os(kernelName);
          os << option.hoistResult.newKernel;
          op->emitOpError()
              << "Found invalid hoist result; op " << *op << " has "
              << op->getNumOperands() << " operands, but hoist result has "
              << option.hoistResult.newInputLayouts.size()
              << " operand layouts for kernel " << kernelName << "\n";
        });
        return UNHOISTABLE;
      }

      hoistingOptions.push_back(option);
    }
  }

  // Select the least costly layout conversion to hoist.
  auto *minHoistingCost = llvm::min_element(
      hoistingOptions, [](const HoistOption &a, const HoistOption &b) {
        return a.cost < b.cost;
      });

  if (minHoistingCost->cost > 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping op, all hoisting results have positive net cost\n");
    return UNHOISTABLE;
  }
  HoistResult minHoistResult = minHoistingCost->hoistResult;

  LLVM_DEBUG(llvm::dbgs()
             << "Minimum cost layout conversion converts output layout from "
             << op->getAttr(kLayoutAttrName) << " to layout "
             << minHoistingCost->hoistResult.newOutputLayout << "\n");

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
                            << minHoistResult.newInputLayouts[i] << "\n");
    LayoutAttr newInputLayout = minHoistResult.newInputLayouts[i];
    auto newInput = ConvertLayoutOp::create(
        builder, op->getLoc(), operand,
        cast<LayoutAttr>(originalLayout.value()), newInputLayout);
    newInput->setAttr(kLayoutAttrName, newInputLayout);
    builder.replaceUsesWithIf(operand, newInput, [&](OpOperand &operand) {
      return operand.getOwner() == op;
    });
  }

  // Set new layout attribute for the op.
  LayoutAttr newOutputLayout = minHoistResult.newOutputLayout;
  op->setAttr(kLayoutAttrName, newOutputLayout);

  // Replace uses of the convert layout op with its input.
  builder.replaceAllUsesWith(minHoistResult.convertLayoutOp.getResult(),
                             minHoistResult.convertLayoutOp.getValue());

  // Ensure that any other convert_layout ops have their from_layouts updated.
  // Update downstream ops.
  for (auto *user : op->getResult(0).getUsers()) {
    if (auto convertLayoutOp = dyn_cast<ConvertLayoutOp>(user)) {
      // Update any convert_layout uses of the op result.
      convertLayoutOp.setFromLayoutAttr(newOutputLayout);
    }
  }

  builder.eraseOp(minHoistResult.convertLayoutOp);
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
        value, ciphertextSize, fromLayout, convertLayoutOp.getToLayout());
    Cost foldedConversion = computeCostOfLayoutConversion(
        value, ciphertextSize, fromLayout, newLayout);
    return OperandChange{fromLayout, newLayout,
                         foldedConversion - originalConversion};
  }
  // Otherwise, we need to insert a new convert_layout op.
  auto originalLayoutResult =
      findAttributeAssociatedWith(value, kLayoutAttrName);
  assert(succeeded(originalLayoutResult) &&
         "Operand does not have a layout attribute");
  auto originalLayout = cast<LayoutAttr>(originalLayoutResult.value());
  return OperandChange{originalLayout, newLayout,
                       computeCostOfLayoutConversion(
                           value, ciphertextSize, originalLayout, newLayout)};
}

Cost LayoutOptimization::costOfChangedResult(Operation *kernel,
                                             LayoutAttr newLayout) {
  Cost totalCost = 0;
  for (auto *user : kernel->getResult(0).getUsers()) {
    if (auto convertLayoutOp = dyn_cast<ConvertLayoutOp>(user)) {
      auto currentValue = convertLayoutOp.getValue();
      Cost originalConversion = computeCostOfLayoutConversion(
          currentValue, ciphertextSize, convertLayoutOp.getFromLayout(),
          convertLayoutOp.getToLayout());
      Cost foldedConversion =
          computeCostOfLayoutConversion(currentValue, ciphertextSize, newLayout,
                                        convertLayoutOp.getToLayout());
      totalCost += foldedConversion - originalConversion;
    }
  }
  return totalCost;
}

static Cost costOfKernelChange(Operation *op, KernelName oldKernel,
                               const HoistResult &hoistResult) {
  // TODO(#1888): add the cost of a kernel change.
  return 0;
}

std::vector<HoistOption> LayoutOptimization::computeHoistingOptions(
    Operation *op, ConvertLayoutOp convertLayoutOp) {
  LayoutConversionHoistableOpInterface hoistableInterface =
      dyn_cast<LayoutConversionHoistableOpInterface>(op);

  if (!hoistableInterface) {
    // TODO(#1597): Add linalg::reduce op kernel.
    LLVM_DEBUG(llvm::dbgs() << "Encountered op with no hoisting interface. "
                               "Returning empty HoistOption\n");
    return {HoistOption()};
  }

  auto hoisters = hoistableInterface.getHoisters(convertLayoutOp);
  LLVM_DEBUG(llvm::dbgs() << "Evaluating " << hoisters.size()
                          << " hoisting options for " << *op << "\n");

  std::vector<HoistResult> results;
  for (auto &hoister : hoisters) {
    auto result = hoister(convertLayoutOp);
    if (failed(result)) {
      continue;
    }
    results.push_back(result.value());
  }

  LLVM_DEBUG(llvm::dbgs() << results.size()
                          << " successful hoisting options\n");

  auto outputLayout = convertLayoutOp.getToLayout();
  KernelAttr oldKernel = op->getAttrOfType<KernelAttr>(kKernelAttrName);
  std::vector<HoistOption> options;
  for (HoistResult &result : results) {
    HoistOption &option = options.emplace_back();
    option.hoistResult = result;
    option.cost = 0;
    // A map is used to deduplicate operand changes.
    DenseMap<std::tuple<Value, LayoutAttr, LayoutAttr>, Cost> operandChangeMap;
    for (auto &operand : op->getOpOperands()) {
      auto computedCost = costOfChangedOperand(operand, op, outputLayout);
      operandChangeMap[std::make_tuple(operand.get(), computedCost.fromLayout,
                                       computedCost.toLayout)] =
          computedCost.cost;
    }
    for (auto &[_, cost] : operandChangeMap) {
      option.cost += cost;
    }
    option.cost += costOfChangedResult(op, outputLayout);
    SmallVector<LayoutAttr> newInputLayouts(op->getNumOperands(), outputLayout);

    // The op may not have a kernel set, in which case the kernel may be trivial
    // and not explicitly marked; in this case we can ignore kernel costs.
    // Otherwise, we can ignore a kernel cost if this hoisting option doesn't
    // change the kernel.
    if ((oldKernel == nullptr &&
         option.hoistResult.newKernel == KernelName::Trivial) ||
        (oldKernel != nullptr &&
         oldKernel.getName() == option.hoistResult.newKernel))
      continue;

    option.cost +=
        costOfKernelChange(op, oldKernel.getName(), option.hoistResult);
  }
  return options;
}

}  // namespace heir
}  // namespace mlir
