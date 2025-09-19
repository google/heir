#include "lib/Transforms/LayoutOptimization/LayoutOptimization.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "lib/Analysis/LayoutFoldingAnalysis/LayoutFoldingAnalysis.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/Kernel.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Transforms/LayoutOptimization/Hoisting.h"
#include "lib/Transforms/LayoutOptimization/LayoutConversionCost.h"
#include "lib/Transforms/LayoutOptimization/Patterns.h"
#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"         // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
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
using ::mlir::heir::tensor_ext::NewLayoutAttr;

constexpr const static StringLiteral kKernelAttrName =
    ::mlir::heir::secret::SecretDialect::kKernelAttrName;

#define GEN_PASS_DEF_LAYOUTOPTIMIZATION
#include "lib/Transforms/LayoutOptimization/LayoutOptimization.h.inc"

namespace {

auto& kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;

using Cost = int64_t;

struct OperandChange {
  Attribute fromLayout;
  Attribute toLayout;
  Cost cost;
};

struct HoistOption {
  // Net increase in cost.
  Cost cost;
  HoistResult hoistResult;
};

}  // namespace

static Cost computeCostOfLayoutConversion(int64_t ciphertextSize,
                                          Attribute fromLayout,
                                          Attribute toLayout) {
  NewLayoutAttr fromLayoutAttr = dyn_cast<NewLayoutAttr>(fromLayout);
  NewLayoutAttr toLayoutAttr = dyn_cast<NewLayoutAttr>(toLayout);

  if (!fromLayoutAttr || !toLayoutAttr) {
    return fromLayout == toLayout ? 0 : 1;
  }

  presburger::IntegerRelation rel = fromLayoutAttr.getIntegerRelation();
  std::optional<int64_t> ctUb =
      rel.getConstantBound64(presburger::BoundType::UB,
                             rel.getVarKindOffset(presburger::VarKind::Range));
  std::optional<int64_t> ctLb =
      rel.getConstantBound64(presburger::BoundType::LB,
                             rel.getVarKindOffset(presburger::VarKind::Range));

  if (!ctUb.has_value() || !ctLb.has_value()) {
    llvm::errs() << "Could not determine number of ciphertexts from layout "
                 << fromLayoutAttr << ", assuming cost 1\n";
    return 1;
  }

  int64_t numCiphertexts = ctUb.value() - ctLb.value() + 1;
  return computeCostOfLayoutConversion(numCiphertexts, ciphertextSize,
                                       fromLayoutAttr, toLayoutAttr);
}

struct LayoutOptimization : impl::LayoutOptimizationBase<LayoutOptimization> {
  using LayoutOptimizationBase::LayoutOptimizationBase;

  enum OpHoistResult { UNHOISTABLE, SUCCESS, FAILURE };
  OpHoistResult hoistOp(Operation* op, IRRewriter& builder,
                        DataFlowSolver* solver);

  std::vector<HoistOption> computeHoistingOptions(
      Operation* op, ConvertLayoutOp convertLayoutOp, DataFlowSolver* solver);

  // Computes cost of changed operand.
  OperandChange costOfChangedOperand(OpOperand& operand, Operation* kernel,
                                     Attribute newLayout,
                                     DataFlowSolver* solver);

  // Computes cost of changed result.
  Cost costOfChangedResult(Operation* kernel, Attribute newLayout);

  void runOnOperation() override;
};

void LayoutOptimization::runOnOperation() {
  auto* ctx = &getContext();
  IRRewriter builder(ctx);

  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<LayoutIsFreeAnalysis>();
  auto solveResult = solver.initializeAndRun(getOperation());

  if (failed(solveResult)) {
    emitError(getOperation()->getLoc(), "Failed to run the analysis.\n");
    signalPassFailure();
    return;
  }

  SmallVector<Operation*> opsToErase;
  WalkResult result =
      getOperation()->walk<WalkOrder::PreOrder, ReverseIterator>(
          [&](Operation* op) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Visiting op: " << op->getName() << " \n");
            if (auto hoistable =
                    dyn_cast<LayoutConversionHoistableOpInterface>(op)) {
              // TODO(#1888): figure out how to get OpInterface verifier to run
              // automatically.
              LLVM_DEBUG(llvm::dbgs() << "Visiting op: " << op->getName()
                                      << " with hoistable interface\n");
              KernelName kernelName = KernelName::Trivial;
              if (op->hasAttr(kKernelAttrName)) {
                kernelName =
                    op->getAttrOfType<KernelAttr>(kKernelAttrName).getName();
                LLVM_DEBUG(llvm::dbgs()
                           << "Op " << op->getName() << " has kernel attribute "
                           << kernelName << "\n");
              } else {
                LLVM_DEBUG(llvm::dbgs()
                           << "Op " << op->getName()
                           << " has no kernel attribute; using trivial\n");
              }

              if (!::mlir::heir::isSupportedKernel(op, kernelName)) {
                op->emitOpError()
                    << "has unsupported kernel: " << kernelName << "\n";
                return WalkResult::interrupt();
              }
            }

            // Attempt to hoist layout conversions before this operation.
            OpHoistResult result = hoistOp(op, builder, &solver);
            if (result == FAILURE) {
              return WalkResult::interrupt();
            };

            // The above may results in sequences of convert_layout ops,
            // or convert_layout ops that occur directly after assign_layout
            // ops, and these can be eagerly folded.
            for (Value value : op->getOperands()) {
              if (auto convertLayoutOp =
                      value.getDefiningOp<ConvertLayoutOp>()) {
                (void)tryFoldLayoutConversionIntoPrevious(
                    builder, convertLayoutOp, opsToErase);
              }
            }
            return WalkResult::advance();
          });

  if (result.wasInterrupted()) {
    signalPassFailure();
  }

  for (auto* op : opsToErase) {
    builder.eraseOp(op);
  }

  RewritePatternSet patterns(ctx);
  patterns.add<HoistArgLayouts, FoldLayoutConversions>(ctx);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
};

LayoutOptimization::OpHoistResult LayoutOptimization::hoistOp(
    Operation* op, IRRewriter& builder, DataFlowSolver* solver) {
  // Folders will canonicalize assign_layout and convert_layout
  if (isa<ConvertLayoutOp, AssignLayoutOp>(op)) {
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
  for (auto* user : op->getResult(0).getUsers()) {
    if (auto convertLayoutOp = dyn_cast<ConvertLayoutOp>(user)) {
      resultLayoutConversions.push_back(convertLayoutOp);
    }
  }

  // No result has a layout conversion directly after.
  if (resultLayoutConversions.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping op, no results followed by convert_layout\n");
    return UNHOISTABLE;
  }

  // Now compute the cost of hoisting each conversion layout.
  SmallVector<HoistOption> hoistingOptions;
  for (auto resultLayoutConversion : resultLayoutConversions) {
    auto options = computeHoistingOptions(op, resultLayoutConversion, solver);
    for (auto& option : options) {
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
  auto* minHoistingCost = llvm::min_element(
      hoistingOptions, [](const HoistOption& a, const HoistOption& b) {
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
    Attribute newInputLayout = minHoistResult.newInputLayouts[i];
    auto newInput = ConvertLayoutOp::create(
        builder, op->getLoc(), operand, originalLayout.value(), newInputLayout);
    newInput->setAttr(kLayoutAttrName, newInputLayout);
    builder.replaceUsesWithIf(operand, newInput, [&](OpOperand& operand) {
      return operand.getOwner() == op;
    });
  }

  // Set new layout attribute for the op.
  Attribute newOutputLayout = minHoistResult.newOutputLayout;
  op->setAttr(kLayoutAttrName, newOutputLayout);

  // Replace uses of the convert layout op with its input.
  builder.replaceAllUsesWith(minHoistResult.convertLayoutOp.getResult(),
                             minHoistResult.convertLayoutOp.getValue());

  // Ensure that any other convert_layout ops have their from_layouts updated.
  // Update downstream ops.
  for (auto* user : op->getResult(0).getUsers()) {
    if (auto convertLayoutOp = dyn_cast<ConvertLayoutOp>(user)) {
      // Update any convert_layout uses of the op result.
      convertLayoutOp.setFromLayoutAttr(newOutputLayout);
    }
  }

  builder.eraseOp(minHoistResult.convertLayoutOp);
  return SUCCESS;
}

OperandChange LayoutOptimization::costOfChangedOperand(OpOperand& operand,
                                                       Operation* kernel,
                                                       Attribute newLayout,
                                                       DataFlowSolver* solver) {
  auto value = operand.get();

  LLVM_DEBUG(llvm::dbgs() << "Checking if operand has free layout: " << value
                          << "\n");
  if (isLayoutFree(value, solver)) {
    LLVM_DEBUG(llvm::dbgs() << "Layout is free!\n");
    return OperandChange{Attribute(), Attribute(), 0};
  }
  LLVM_DEBUG(llvm::dbgs() << "Layout is not free\n");

  if (auto convertLayoutOp = value.getDefiningOp<ConvertLayoutOp>()) {
    // If the operand came from convert_layout, the cost of the change is
    // (folded conversion - original conversion).
    auto fromLayout = convertLayoutOp.getFromLayout();
    Cost originalConversion = computeCostOfLayoutConversion(
        ciphertextSize, fromLayout, convertLayoutOp.getToLayout());
    Cost foldedConversion =
        computeCostOfLayoutConversion(ciphertextSize, fromLayout, newLayout);
    return OperandChange{fromLayout, newLayout,
                         foldedConversion - originalConversion};
  }

  // Otherwise, we need to insert a new convert_layout op.
  auto originalLayoutResult =
      findAttributeAssociatedWith(value, kLayoutAttrName);
  assert(succeeded(originalLayoutResult) &&
         "Operand does not have a layout attribute");
  auto originalLayout = originalLayoutResult.value();
  return OperandChange{
      originalLayout, newLayout,
      computeCostOfLayoutConversion(ciphertextSize, originalLayout, newLayout)};
}

Cost LayoutOptimization::costOfChangedResult(Operation* kernel,
                                             Attribute newLayout) {
  Cost totalCost = 0;
  for (auto* user : kernel->getResult(0).getUsers()) {
    if (auto convertLayoutOp = dyn_cast<ConvertLayoutOp>(user)) {
      Cost originalConversion = computeCostOfLayoutConversion(
          ciphertextSize, convertLayoutOp.getFromLayout(),
          convertLayoutOp.getToLayout());
      Cost foldedConversion = computeCostOfLayoutConversion(
          ciphertextSize, newLayout, convertLayoutOp.getToLayout());
      totalCost += foldedConversion - originalConversion;
    }
  }
  return totalCost;
}

static Cost costOfKernelChange(Operation* op, KernelName oldKernel,
                               const HoistResult& hoistResult) {
  // TODO(#1888): add the cost of a kernel change.
  return 0;
}

std::vector<HoistOption> LayoutOptimization::computeHoistingOptions(
    Operation* op, ConvertLayoutOp convertLayoutOp, DataFlowSolver* solver) {
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
  for (auto& hoister : hoisters) {
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
  for (HoistResult& result : results) {
    HoistOption& option = options.emplace_back();
    option.hoistResult = result;
    option.cost = 0;

    LLVM_DEBUG(llvm::dbgs()
               << "Computing cost of hoisting layout " << result.newOutputLayout
               << " via kernel " << result.newKernel << "\n");

    // A map is used to deduplicate operand changes.
    DenseMap<std::tuple<Value, Attribute, Attribute>, Cost> operandChangeMap;
    SmallVector<Cost> operandChangeCosts;
    for (auto& operand : op->getOpOperands()) {
      auto computedCost =
          costOfChangedOperand(operand, op, outputLayout, solver);
      operandChangeCosts.push_back(computedCost.cost);
      auto key = std::make_tuple(operand.get(), computedCost.fromLayout,
                                 computedCost.toLayout);
      operandChangeMap[key] = computedCost.cost;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "\toperand change costs (with duplicates): ";
      for (auto cost : operandChangeCosts) {
        llvm::dbgs() << cost << ", ";
      }
      llvm::dbgs() << "\n";
    });

    Cost totalOperandChangeCost = 0;
    for (auto& [_, cost] : operandChangeMap) {
      totalOperandChangeCost += cost;
    }
    LLVM_DEBUG(llvm::dbgs() << "\toperand change total cost: "
                            << totalOperandChangeCost << "\n");
    option.cost += totalOperandChangeCost;

    Cost resultChangeCost = costOfChangedResult(op, outputLayout);
    LLVM_DEBUG(llvm::dbgs()
               << "\tresult change cost: " << resultChangeCost << "\n");
    option.cost += resultChangeCost;

    // The op may not have a kernel set, in which case the kernel may be
    // trivial and not explicitly marked; in this case we can ignore kernel
    // costs. Otherwise, we can ignore a kernel cost if this hoisting option
    // doesn't change the kernel.
    if ((oldKernel == nullptr &&
         option.hoistResult.newKernel == KernelName::Trivial) ||
        (oldKernel != nullptr &&
         oldKernel.getName() == option.hoistResult.newKernel))
      continue;

    Cost kernelChangeCost =
        costOfKernelChange(op, oldKernel.getName(), option.hoistResult);
    LLVM_DEBUG(llvm::dbgs()
               << "\tkernel change cost: " << kernelChangeCost << "\n");
    option.cost += kernelChangeCost;
  }
  return options;
}

}  // namespace heir
}  // namespace mlir
