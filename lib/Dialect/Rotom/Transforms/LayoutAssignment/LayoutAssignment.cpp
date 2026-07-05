#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/LayoutAssignment.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/AssignmentContext.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/CostModel.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/DimMaps.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Generators.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/ValueUtils.h"
#include "lib/Dialect/Rotom/Utils/ContractionAlignment.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"        // from @llvm-project

namespace mlir::heir::rotom {

constexpr llvm::StringLiteral kRotomSeedAttrName = "rotom.seed";
constexpr llvm::StringLiteral kRotomLayoutAttrName = "rotom.layout";

#define DEBUG_TYPE "rotom-assign-layout"

#define GEN_PASS_DEF_LAYOUTASSIGNMENT
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/LayoutAssignment.h.inc"

namespace {
struct LayoutAssignment : public impl::LayoutAssignmentBase<LayoutAssignment>,
                          public AssignmentContext {
  using LayoutAssignmentBase::LayoutAssignmentBase;

  DenseMap<Value, SmallVector<Candidate>> candidates;
  DenseMap<Value, LayoutAttr> selectedLayouts;
  DenseMap<std::pair<Attribute, Attribute>, int64_t> conversionCostCache;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<secret::SecretDialect>();
    registry.insert<tensor_ext::TensorExtDialect>();
  }

  void runOnOperation() override;

  // --- Candidate generation: a forward pass that fills `candidates` for every
  // value. visitOperation dispatches to one free generate* function per tensor
  // op (declared in Generators.h, grouped by op family under gen/); each takes
  // only an AssignmentContext, so the per-op kernel space can grow file by
  // file. The methods here implement that AssignmentContext -- the shared
  // seed/candidate/cost API the generators call back into. ---
  LogicalResult generateCandidates(ModuleOp module);
  LogicalResult visitOperation(Operation* op);
  void seedValue(Value value) override;
  void setCandidates(Value value, ArrayRef<Candidate> newCandidates) override;
  SmallVector<Candidate> candidatesForValue(Value value) override;
  void assignResultsFromCandidates(Operation* op,
                                   ArrayRef<Candidate> chosen) override;
  SmallVector<Candidate> chooseCommonOperandCandidates(
      Operation* op, KernelKind kind) override;
  SmallVector<Candidate> chooseElementwiseKernels(
      ArrayRef<Value> operands, KernelKind kind,
      function_ref<int64_t(LayoutAttr)> computeCostFn,
      std::optional<KernelName> rotomKernel) override;
  int64_t cachedConversionCost(LayoutAttr from, LayoutAttr to) override;

  // --- Layout search: each value's candidates already carry the dedup'd
  // accumulated cost and the full assignment of everything feeding it, so
  // selection is just picking the cheapest consistent assignment at each
  // function's returned values -- no backward propagation. ---
  void selectLayouts(ModuleOp module);
  LogicalResult visitReturn(func::ReturnOp op);
  void applyKernels(ModuleOp module);
  void writeSelectedLayouts();
};
}  // namespace

void LayoutAssignment::seedValue(Value value) {
  if (candidates.contains(value)) return;

  FailureOr<Attribute> seedAttr =
      findAttributeAssociatedWith(value, kRotomSeedAttrName);
  if (failed(seedAttr)) return;

  auto seed = dyn_cast<SeedAttr>(*seedAttr);
  if (!seed) return;

  SmallVector<Candidate> seeded;
  for (Attribute attr : seed.getLayouts()) {
    auto layout = dyn_cast<LayoutAttr>(attr);
    if (!layout) continue;
    Candidate seed;
    seed.layout = layout;
    seed.kind = KernelKind::Tensor;
    seeded.push_back(std::move(seed));
  }
  if (!seeded.empty()) setCandidates(value, seeded);
}

void LayoutAssignment::setCandidates(Value value,
                                     ArrayRef<Candidate> newCandidates) {
  if (!isTensorLike(value) || newCandidates.empty()) return;
  SmallVector<Candidate> compatibleCandidates;
  for (const Candidate& candidate : newCandidates) {
    if (!isLayoutCompatibleWithValue(candidate.layout, value)) continue;
    // Fold this value's own kernel into its assignment, so the assignment is
    // the complete assignment of the value and everything feeding it, and
    // `cost` is the dedup'd sum over that assignment.
    Candidate finalized = candidate;
    finalized.assignment[value] = {finalized.layout, finalized.localCost};
    finalized.accumulatedCost = accumulatedCostOf(finalized.assignment);
    compatibleCandidates.push_back(std::move(finalized));
  }
  if (compatibleCandidates.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No layout candidate is compatible with value "
                            << value << "\n");
    return;
  }

  candidates[value] = uniqueCandidates(compatibleCandidates);
  LLVM_DEBUG({
    llvm::dbgs() << "Assigned " << candidates[value].size()
                 << " candidate(s) to value " << value << "\n";
  });
}

SmallVector<Candidate> LayoutAssignment::candidatesForValue(Value value) {
  seedValue(value);
  auto it = candidates.find(value);
  if (it == candidates.end()) return {};
  return it->second;
}

SmallVector<Candidate> LayoutAssignment::chooseCommonOperandCandidates(
    Operation* op, KernelKind kind) {
  SmallVector<Value> operands;
  SmallVector<SmallVector<Candidate>> candidateSets;
  for (Value operand : op->getOperands()) {
    if (!isTensorLike(operand)) continue;
    SmallVector<Candidate> operandCandidates = candidatesForValue(operand);
    if (operandCandidates.empty()) continue;
    operands.push_back(operand);
    candidateSets.push_back(operandCandidates);
  }
  return chooseCommonCandidates(
      operands, candidateSets, kind,
      [&](LayoutAttr layout) { return operationCost(op, layout); },
      [this](LayoutAttr from, LayoutAttr to) {
        return cachedConversionCost(from, to);
      });
}

int64_t LayoutAssignment::cachedConversionCost(LayoutAttr from, LayoutAttr to) {
  if (from == to) return 0;
  // Cheap fast path: aligned layouts (ciphertext-order differences are free)
  // need no rotations, so skip the expensive shift-network estimate.
  if (conversionMoves(from, to).empty()) return 0;

  std::pair<Attribute, Attribute> key(from, to);
  auto it = conversionCostCache.find(key);
  if (it != conversionCostCache.end()) return it->second;

  const RotomCostModel& costModel = getCostModel();
  int64_t cost;
  if (layoutNumCiphertexts(from) == layoutNumCiphertexts(to)) {
    // Real Vos-Vos-Erkin rotation count weighted by the per-rotation cost.
    // Every candidate layout is materializable (the search only generates such
    // layouts), so the shift network always yields a cost.
    std::optional<int64_t> rotations = shiftNetworkConversionCost(from, to);
    assert(rotations &&
           "shift network must yield a cost for assignable layouts");
    cost = *rotations * costModel.rotation;
  } else {
    // Ciphertext-count-changing conversion (expansion or compaction), which
    // tensor_ext.convert_layout cannot express: price the explicit
    // rotate/mask/accumulate steps exactly as the lowering emits them, so
    // expensive conversions lose in the search on cost, not on capability.
    FailureOr<SmallVector<LayoutExpansionStep>> steps =
        planLayoutExpansion(from, to);
    assert(succeeded(steps) &&
           "layout expansion must plan for assignable layouts");
    const int64_t n = from.getN();
    cost = 0;
    DenseSet<int64_t> targetsSeen;
    for (const LayoutExpansionStep& step : *steps) {
      if (step.shift != 0) cost += costModel.rotation;
      // A partial-row step needs a plaintext mask multiply (cheap; priced as
      // an add pending a dedicated plaintext-multiply weight).
      if (static_cast<int64_t>(step.targetSlots.size()) != n) {
        cost += costModel.add;
      }
      if (!targetsSeen.insert(step.targetCt).second) cost += costModel.add;
    }
  }
  // Cached per layout pair since the search queries the same pairs across many
  // candidate pairings.
  conversionCostCache[key] = cost;
  return cost;
}

SmallVector<Candidate> LayoutAssignment::chooseElementwiseKernels(
    ArrayRef<Value> operands, KernelKind kind,
    function_ref<int64_t(LayoutAttr)> computeCostFn,
    std::optional<KernelName> rotomKernel) {
  if (operands.size() != 2) return {};

  auto lhsType =
      dyn_cast<RankedTensorType>(getPlainValueType(operands[0].getType()));
  auto rhsType =
      dyn_cast<RankedTensorType>(getPlainValueType(operands[1].getType()));
  if (!lhsType || !rhsType || lhsType.getRank() != rhsType.getRank()) {
    return {};
  }

  SmallVector<Candidate> lhsCandidates = candidatesForValue(operands[0]);
  SmallVector<Candidate> rhsCandidates = candidatesForValue(operands[1]);
  SmallVector<Value> operandValues(operands.begin(), operands.end());

  // Each (lhs, rhs) candidate pairing yields up to two kernels: compute at the
  // lhs layout (converting rhs onto it) or at the rhs layout. The kernel's
  // local cost is the compute plus the slot-rotation cost of aligning both
  // operands onto the shared compute layout (one of which is free). The
  // operands' assignments are merged, so a pairing whose sub-assignments
  // disagree on a shared value is dropped, and the merge never double-counts
  // shared work.
  SmallVector<Candidate> kernels;
  for (const Candidate& lhs : lhsCandidates) {
    for (const Candidate& rhs : rhsCandidates) {
      Assignment merged;
      if (!mergeAssignments(merged, lhs.assignment) ||
          !mergeAssignments(merged, rhs.assignment)) {
        continue;
      }
      auto addKernel = [&](LayoutAttr computeLayout) {
        int64_t localCost = computeCostFn(computeLayout) +
                            cachedConversionCost(lhs.layout, computeLayout) +
                            cachedConversionCost(rhs.layout, computeLayout);
        std::optional<KernelName> kernel;
        if (rotomKernel && supportsRotomAlignmentLowering(
                               lhs.layout, rhs.layout, computeLayout)) {
          kernel = rotomKernel;
        }
        Candidate candidate;
        candidate.layout = computeLayout;
        candidate.kind = kind;
        candidate.operands = operandValues;
        candidate.operandLayouts = {lhs.layout, rhs.layout};
        candidate.kernel = kernel;
        candidate.localCost = localCost;
        candidate.assignment = merged;
        candidate.accumulatedCost =
            accumulatedCostOf(candidate.assignment) + candidate.localCost;
        kernels.push_back(std::move(candidate));
      };
      addKernel(lhs.layout);  // compute at lhs, convert rhs onto it
      if (rhs.layout != lhs.layout) {
        addKernel(rhs.layout);  // compute at rhs, convert lhs onto it
      }
    }
  }
  return uniqueCandidates(kernels);
}

void LayoutAssignment::assignResultsFromCandidates(Operation* op,
                                                   ArrayRef<Candidate> chosen) {
  if (chosen.empty()) return;
  for (Value result : op->getResults()) {
    if (!isTensorLike(result)) continue;
    setCandidates(result, chosen);
  }
}

LogicalResult LayoutAssignment::visitReturn(func::ReturnOp op) {
  auto func = op->getParentOfType<func::FuncOp>();

  // The returned values share one function-wide assignment. Fold each return
  // operand's cheapest candidate whose assignment is consistent with the
  // choices made so far into a single assignment; that assignment is the
  // function's layout assignment.
  Assignment assignment;
  SmallVector<std::pair<unsigned, LayoutAttr>> resultLayouts;
  for (OpOperand& operand : op->getOpOperands()) {
    Value value = operand.get();
    seedValue(value);
    auto it = candidates.find(value);
    if (it == candidates.end() || it->second.empty()) continue;
    for (const Candidate& candidate : it->second) {  // cheapest first
      Assignment trial = assignment;
      if (mergeAssignments(trial, candidate.assignment)) {
        assignment = std::move(trial);
        resultLayouts.push_back({operand.getOperandNumber(), candidate.layout});
        break;
      }
    }
  }

  for (const auto& entry : assignment) {
    selectedLayouts[entry.first] = entry.second.first;
  }
  for (const auto& [index, layout] : resultLayouts) {
    func.setResultAttr(index, kRotomLayoutAttrName, layout);
  }
  return success();
}

LogicalResult LayoutAssignment::visitOperation(Operation* op) {
  AssignmentContext& ctx = *this;
  return TypeSwitch<Operation*, LogicalResult>(op)
      .Case<func::FuncOp>(
          [&](auto typedOp) { return generateFunc(ctx, typedOp); })
      .Case<secret::GenericOp>(
          [&](auto typedOp) { return generateSecretGeneric(ctx, typedOp); })
      .Case<secret::YieldOp>(
          [&](auto typedOp) { return generateYield(ctx, typedOp); })
      .Case<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
            arith::MulFOp, arith::MulIOp>(
          [&](auto typedOp) { return generateElementwise(ctx, typedOp); })
      .Case<linalg::GenericOp>(
          [&](auto typedOp) { return generateLinalgGeneric(ctx, typedOp); })
      .Case<linalg::MatmulOp>(
          [&](auto typedOp) { return generateMatmul(ctx, typedOp); })
      .Case<linalg::TransposeOp>(
          [&](auto typedOp) { return generateTranspose(ctx, typedOp); })
      .Case<linalg::ReduceOp>(
          [&](auto typedOp) { return generateReduction(ctx, typedOp); })
      .Case<tensor::CollapseShapeOp>(
          [&](auto typedOp) { return generateCollapseShape(ctx, typedOp); })
      .Case<tensor::ExpandShapeOp>(
          [&](auto typedOp) { return generateExpandShape(ctx, typedOp); })
      .Case<tensor::ExtractSliceOp>(
          [&](auto typedOp) { return generateExtractSlice(ctx, typedOp); })
      .Case<tensor::InsertSliceOp>(
          [&](auto typedOp) { return generateInsertSlice(ctx, typedOp); })
      .Default([&](Operation* genericOp) {
        return generatePassThrough(ctx, genericOp);
      });
}

void LayoutAssignment::writeSelectedLayouts() {
  for (auto& [value, layout] : selectedLayouts) {
    setAttributeAssociatedWith(value, kRotomLayoutAttrName, layout);
  }
}

LogicalResult LayoutAssignment::generateCandidates(ModuleOp module) {
  // Forward pre-order walk: every op contributes candidate layouts for its
  // results. Returns are search roots, not generators, so they are skipped.
  WalkResult result = module.walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (isa<func::ReturnOp>(op)) return WalkResult::advance();
    if (failed(visitOperation(op))) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void LayoutAssignment::selectLayouts(ModuleOp module) {
  // Each function's returned values' candidates already carry the full cost and
  // assignment of their inputs, so selection is just committing the cheapest
  // consistent assignment per function.
  module.walk([&](func::ReturnOp op) { (void)visitReturn(op); });
}

void LayoutAssignment::applyKernels(ModuleOp module) {
  // The assignment carries layouts, not kernels. Re-derive each elementwise
  // op's Rotom kernel attribute from the final operand/result layouts (the only
  // ops that carry one); a forced kernel is left untouched.
  module.walk([&](Operation* op) {
    if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
      // A matmul gets no kernel: its lowering is a pure function of the
      // (lhs, rhs, result) layout combination. Record those layouts on the
      // op, since the materializer erases the per-value rotom attributes
      // before the ciphertext lowering re-derives the plan. Several plans
      // can share a result layout (a rolled plan and its roll-free sibling),
      // so also record the priced winner's computeLayout -- the plan's
      // unique identity -- as a fourth element.
      LayoutAttr lhs = selectedLayouts.lookup(matmul.getInputs()[0]);
      LayoutAttr rhs = selectedLayouts.lookup(matmul.getInputs()[1]);
      LayoutAttr result = selectedLayouts.lookup(matmul->getResult(0));
      if (lhs && rhs && result) {
        SmallVector<Attribute> recorded = {lhs, rhs, result};
        if (std::optional<MatmulPlan> plan =
                selectMatmulPlan(*this, lhs, rhs, result)) {
          recorded.push_back(plan->computeLayout);
        }
        matmul->setAttr(kRotomMatmulAttrName,
                        ArrayAttr::get(matmul.getContext(), recorded));
      }
      return;
    }
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) return;
    if (!isAddLike(op) && !isMulLike(op)) return;
    auto existing = op->getAttrOfType<secret::KernelAttr>(
        secret::SecretDialect::kKernelAttrName);
    if (existing && existing.getForce()) return;

    std::optional<KernelName> kernel = selectRotomElementwiseKernel(op);
    LayoutAttr lhs = selectedLayouts.lookup(op->getOperand(0));
    LayoutAttr rhs = selectedLayouts.lookup(op->getOperand(1));
    LayoutAttr result = selectedLayouts.lookup(op->getResult(0));
    if (kernel && lhs && rhs && result &&
        supportsRotomAlignmentLowering(lhs, rhs, result)) {
      op->setAttr(secret::SecretDialect::kKernelAttrName,
                  secret::KernelAttr::get(op->getContext(), *kernel,
                                          /*force=*/false));
    } else if (existing) {
      op->removeAttr(secret::SecretDialect::kKernelAttrName);
    }
  });
}

void LayoutAssignment::runOnOperation() {
  ModuleOp module = getOperation();
  if (failed(generateCandidates(module))) {
    signalPassFailure();
    return;
  }
  selectLayouts(module);
  applyKernels(module);
  writeSelectedLayouts();
}

}  // namespace mlir::heir::rotom
