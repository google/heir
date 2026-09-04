#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/LayoutAssignment.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/IR/RotomOps.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/AssignmentContext.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/CostModel.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/DimMaps.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Generators.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/ValueUtils.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Dialect/Rotom/Utils/LayoutConversion.h"
#include "lib/Dialect/Rotom/Utils/RotomLayout.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
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
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"        // from @llvm-project

namespace mlir::heir::rotom {

constexpr llvm::StringLiteral kRotomSeedAttrName = "rotom.seed";
constexpr llvm::StringLiteral kRotomLayoutAttrName = "rotom.layout";

#define GEN_PASS_DEF_LAYOUTASSIGNMENT
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/LayoutAssignment.h.inc"

namespace {
// The legality gate for a two-operand kernel at `result`: every layout
// lowerable, all at one ciphertext size.
bool lowerableTriple(LayoutAttr lhs, LayoutAttr rhs, LayoutAttr result) {
  return lhs && rhs && result && lhs.getN() == rhs.getN() &&
         lhs.getN() == result.getN() && isLowerableRotomLayout(lhs) &&
         isLowerableRotomLayout(rhs) && isLowerableRotomLayout(result);
}
struct LayoutAssignment : public impl::LayoutAssignmentBase<LayoutAssignment>,
                          public AssignmentContext {
  using LayoutAssignmentBase::LayoutAssignmentBase;

  DenseMap<Value, SmallVector<Candidate>> candidates;
  DenseMap<Value, LayoutAttr> selectedLayouts;

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
      bool hasRotomKernel) override;
  int64_t conversionCost(LayoutAttr from, LayoutAttr to) override;
  int64_t cacheConversionCost(LayoutAttr from, LayoutAttr to, int64_t cost);
  llvm::DenseMap<std::pair<LayoutAttr, LayoutAttr>, int64_t>
      conversionCostCache;

  // --- Layout search: each value's candidates already carry the dedup'd
  // accumulated cost and the full assignment of everything feeding it, so
  // selection is just picking the cheapest consistent assignment at each
  // function's returned values -- no backward propagation. ---
  void selectLayouts(ModuleOp module);
  LogicalResult visitReturn(func::ReturnOp op);
  void applyKernels(ModuleOp module);
  void writeSelectedLayouts();

  // --- Kernel materialization: the selected matmul plans become IR. Each
  // rewrite emits the operand alignment ops (rotom.convert_layout and
  // rotom.apply_roll) and one rotom.matmul carrying every layout of the
  // plan, then erases the linalg.matmul and its dead zero-init chain. ---
  struct MatmulRewrite {
    linalg::MatmulOp op;
    MatmulAlignment alignment;
  };
  SmallVector<MatmulRewrite> matmulRewrites;
  // Emits the chain as ops: a rotom.apply_roll for a step that adds one roll,
  // a rotom.convert_layout otherwise.
  Value alignOperand(OpBuilder& builder, Location loc, Value value,
                     ArrayRef<LayoutAttr> chain);
  void rewriteMatmul(const MatmulRewrite& rewrite);
};
}  // namespace

// A public constant is data packed at encode time, so each use may pack it
// at the layout its own consumer wants (the reference's one-term-per-literal
// model). CSE merges equal literals into one op; split them back per use so
// two consumers never fight over one layout.
static void splitConstantsPerUse(Operation* root) {
  SmallVector<arith::ConstantOp> shared;
  root->walk([&](arith::ConstantOp op) {
    if (!isa<RankedTensorType>(op.getType())) return;
    if (!op->hasOneUse() && !op->use_empty()) shared.push_back(op);
  });
  for (arith::ConstantOp op : shared) {
    SmallVector<OpOperand*> uses;
    for (OpOperand& use : op->getUses()) uses.push_back(&use);
    // The first use keeps the original; every other use gets its own copy,
    // placed right after the original: same block, so the boundary
    // materialization treats it exactly like the original, and it dominates
    // every use the original did.
    for (size_t i = 1; i < uses.size(); ++i) {
      OpBuilder builder(op.getOperation());
      builder.setInsertionPointAfter(op);
      Operation* clone = builder.clone(*op.getOperation());
      uses[i]->set(clone->getResult(0));
    }
  }
}

void LayoutAssignment::seedValue(Value value) {
  if (candidates.contains(value)) return;

  FailureOr<Attribute> seedAttr =
      findAttributeAssociatedWith(value, kRotomSeedAttrName);
  if (failed(seedAttr)) return;

  auto seed = dyn_cast<SeedAttr>(*seedAttr);
  if (!seed) return;

  SmallVector<Candidate> seeded;
  auto addSourceCandidate = [&](LayoutAttr layout) {
    Candidate seed;
    seed.layout = layout;
    seed.kind = KernelKind::Tensor;
    seeded.push_back(std::move(seed));
  };
  // Seeds are offered as-is: rolled placements are not blanket-seeded. A
  // consumer that wants a diagonal packing of a source introduces it itself
  // -- a matmul plan repacks the source at its expanded placement (see
  // alignOptions), and anything else pays the priced conversion.
  for (Attribute attr : seed.getLayouts()) {
    auto layout = dyn_cast<LayoutAttr>(attr);
    if (!layout) continue;
    addSourceCandidate(layout);
  }
  if (!seeded.empty()) setCandidates(value, seeded);
}

void LayoutAssignment::setCandidates(Value value,
                                     ArrayRef<Candidate> newCandidates) {
  if (!isTensorLike(value) || newCandidates.empty()) return;
  const RotomCostModel& costModel = getCostModel();
  SmallVector<Candidate> compatibleCandidates;
  for (const Candidate& candidate : newCandidates) {
    if (!isLayoutCompatibleWithValue(candidate.layout, value)) continue;
    // Fold this value's own kernel into its assignment, so the assignment is
    // the complete assignment of the value and everything feeding it, and
    // `cost` is the dedup'd sum over that assignment.
    Candidate finalized = candidate;
    // Charge the ciphertext-count carrying cost of holding this value at
    // this layout (a compactness pressure; see RotomCostModel). Block
    // arguments and yields alias a value already charged, so they are
    // exempt.
    int64_t carryingCost = 0;
    if (finalized.kind != KernelKind::BlockArgument &&
        finalized.kind != KernelKind::Yield) {
      carryingCost =
          costModel.ciphertextCount * layoutNumCiphertexts(finalized.layout);
    }
    finalized.assignment[value] = {finalized.layout,
                                   finalized.localCost + carryingCost};
    finalized.accumulatedCost = accumulatedCostOf(finalized.assignment);
    compatibleCandidates.push_back(std::move(finalized));
  }
  if (compatibleCandidates.empty()) {
    return;
  }

  candidates[value] = uniqueCandidates(compatibleCandidates);
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
        return conversionCost(from, to);
      });
}

int64_t LayoutAssignment::conversionCost(LayoutAttr from, LayoutAttr to) {
  if (from == to) return 0;
  // The estimate counts the step plan, so it is worth remembering: the search
  // asks for the same pair many times.
  auto cached = conversionCostCache.find({from, to});
  if (cached != conversionCostCache.end()) return cached->second;

  // Structure-level estimate, O(#pieces): the search prices conversions from
  // the layouts' traversal-piece and roll structures (batched distinct-shift
  // rotations, one multiplicative depth of masking) without materializing
  // relations or points, so search time does not scale with tensor extents
  // or ciphertext counts. The emission derives the exact step plan from the
  // relations once per SELECTED conversion; the estimate matches its counts
  // on the structured conversions the search
  // favors (fills, diagonalizations, block moves).
  const RotomCostModel& costModel = getCostModel();
  ConversionEstimate estimate = estimateConversionCost(from, to);
  if (!estimate.lowerable) {
    return cacheConversionCost(from, to, kUnlowerableConversion);
  }
  return cacheConversionCost(
      from, to,
      estimate.rotations * costModel.rotation +
          (estimate.masks + estimate.accumulates) * costModel.add);
}

int64_t LayoutAssignment::cacheConversionCost(LayoutAttr from, LayoutAttr to,
                                              int64_t cost) {
  conversionCostCache[{from, to}] = cost;
  return cost;
}

// A value is public when nothing it derives from is secret. Its data is known
// at encode time, so it can be packed at any layout the consumer wants (the
// reference's `not layout.secret` test, which replaces the operand's kernel
// with a TENSOR kernel).
bool isPublicValue(Value value) {
  llvm::DenseSet<Value> seen;
  SmallVector<Value> worklist = {value};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!seen.insert(current).second) continue;
    if (isa<secret::SecretType>(current.getType())) return false;
    if (auto arg = dyn_cast<BlockArgument>(current)) {
      // A secret.generic body argument carries the secret operand's data.
      if (isa<secret::GenericOp>(arg.getOwner()->getParentOp())) return false;
      continue;
    }
    Operation* def = current.getDefiningOp();
    if (!def) return false;
    worklist.append(def->operand_begin(), def->operand_end());
  }
  return true;
}

// The near-zero price of repacking a source at a demanded placement: enough
// to lose ties against an operand that already sits at its seeded layout, far
// below any ciphertext conversion. Mirrors gen/Binop.cpp's matmul path.
constexpr int64_t kSourceRepackCost = 1;

// `oldToNew` (total over the source's dims, -1 to drop) read backwards:
// total over the result's dims, -1 for the axes the op created.
static SmallVector<int64_t> invertDimMap(ArrayRef<int64_t> oldToNew,
                                         int64_t newRank) {
  SmallVector<int64_t> newToOld(newRank, -1);
  for (auto [oldDim, newDim] : llvm::enumerate(oldToNew)) {
    if (newDim < 0 || newDim >= newRank) continue;
    // A collapse folds several source dims onto one result dim, so the map
    // has no unambiguous inverse; the caller must not repack through it.
    if (newToOld[newDim] >= 0) return {};
    newToOld[newDim] = static_cast<int64_t>(oldDim);
  }
  return newToOld;
}

// Repacks `value` at `layout` by carrying the layout back to the value the
// client interface actually packs. Every step between is a pure relabel, so
// the layout is remapped through the step's dim map rather than converted.
// Returns false when the chain cannot carry it -- a step this does not
// model, or a value with another consumer whose layout would be clobbered --
// and then the candidate must be dropped, leaving the priced conversion
// variants to compete.
static bool repackPublicChain(Value value, LayoutAttr layout,
                              Assignment& assignment,
                              const RotomCostModel& costModel) {
  Value current = value;
  LayoutAttr currentLayout = layout;
  // A packing chain is a handful of reshapes; the bound just stops a cycle.
  for (int step = 0; step < 8; ++step) {
    if (!current.hasOneUse()) return false;
    assignment[current] = {
        currentLayout,
        kSourceRepackCost +
            costModel.ciphertextCount * layoutNumCiphertexts(currentLayout)};
    Operation* def = current.getDefiningOp();
    if (!def) return true;  // a block argument: what the client interface packs

    SmallVector<int64_t> newToOld;
    Value source;
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(def)) {
      std::optional<SmallVector<int64_t>> forward = getExpandShapeDimMap(
          expand.getResultType(), expand.getReassociationIndices());
      if (!forward) return false;
      newToOld = invertDimMap(*forward, expand.getResultType().getRank());
      source = expand.getSrc();
    } else if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(def)) {
      std::optional<SmallVector<int64_t>> forward = getCollapseShapeDimMap(
          collapse.getSrcType(), collapse.getReassociationIndices());
      if (!forward) return false;
      newToOld = invertDimMap(*forward, collapse.getResultType().getRank());
      source = collapse.getSrc();
    } else if (auto transpose = dyn_cast<linalg::TransposeOp>(def)) {
      // Result dim i reads source dim permutation[i].
      newToOld = SmallVector<int64_t>(transpose.getPermutation());
      source = transpose.getInput();
    } else if (def->hasTrait<OpTrait::ConstantLike>()) {
      return true;  // a constant is packed where it stands
    } else {
      return false;
    }
    if (newToOld.empty()) return false;
    LayoutAttr pre = remapLayoutDims(currentLayout, newToOld);
    if (!pre) return false;
    current = source;
    currentLayout = pre;
  }
  return false;
}

SmallVector<Candidate> LayoutAssignment::chooseElementwiseKernels(
    ArrayRef<Value> operands, KernelKind kind,
    function_ref<int64_t(LayoutAttr)> computeCostFn, bool hasRotomKernel) {
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

  // Each (lhs, rhs) candidate pairing is aligned by the shared engine under
  // the identity map: an aligned pair stands, otherwise one side converts
  // onto the other's structure. The kernel computes at the aligned layout;
  // its local cost is the compute plus the estimated cost of bringing each
  // operand there (one side is free). The operands' assignments are merged,
  // so a pairing whose sub-assignments disagree on a shared value is dropped,
  // and the merge never double-counts shared work.
  const auto map = OperatorAlignmentMap::identity(lhsType.getRank());
  SmallVector<Candidate> kernels;
  for (const Candidate& lhs : lhsCandidates) {
    for (const Candidate& rhs : rhsCandidates) {
      Assignment merged;
      if (!mergeAssignments(merged, lhs.assignment) ||
          !mergeAssignments(merged, rhs.assignment)) {
        continue;
      }
      for (const AlignedPair& aligned :
           alignPair(map, lhs.layout, rhs.layout)) {
        if (aligned.lhs != aligned.rhs) continue;  // elementwise: one layout
        LayoutAttr computeLayout = aligned.lhs;
        // A source is data packed at encode time, so it can be packed at the
        // compute layout instead of converted to it (the reference's
        // match_public_kernel, which every binop path calls). Without this
        // choice a ciphertext is dragged to the constant's seeded layout,
        // and the conversion costs far more than repacking the constant.
        const bool lhsRepackable = isPublicValue(operands[0]) &&
                                   lhs.layout != computeLayout &&
                                   isLowerableRotomLayout(computeLayout);
        const bool rhsRepackable = isPublicValue(operands[1]) &&
                                   rhs.layout != computeLayout &&
                                   isLowerableRotomLayout(computeLayout);
        for (int variant = 0; variant < 4; ++variant) {
          const bool repackLhs = (variant & 1) && lhsRepackable;
          const bool repackRhs = (variant & 2) && rhsRepackable;
          if (repackLhs != bool(variant & 1)) continue;
          if (repackRhs != bool(variant & 2)) continue;
          // One value, one layout: a self-op must agree with itself.
          if (operands[0] == operands[1] && repackLhs != repackRhs) continue;
          LayoutAttr lhsLayout = repackLhs ? computeLayout : lhs.layout;
          LayoutAttr rhsLayout = repackRhs ? computeLayout : rhs.layout;
          int64_t localCost =
              computeCostFn(computeLayout) +
              (repackLhs ? 0 : conversionCost(lhs.layout, computeLayout)) +
              (repackRhs ? 0 : conversionCost(rhs.layout, computeLayout));
          const bool kernel =
              hasRotomKernel &&
              lowerableTriple(lhsLayout, rhsLayout, computeLayout);
          // A candidate must have an emission path: the Rotom alignment
          // kernel, or all three layouts coinciding (pointwise compute at any
          // layout). A pairing that needs alignment the kernel cannot lower
          // is dropped, so the selection walks back to producer candidates.
          if (!kernel &&
              !(lhsLayout == computeLayout && rhsLayout == computeLayout)) {
            continue;
          }
          Candidate candidate;
          candidate.layout = computeLayout;
          candidate.kind = kind;
          candidate.operands = operandValues;
          candidate.operandLayouts = {lhsLayout, rhsLayout};
          candidate.hasRotomKernel = kernel;
          candidate.localCost = localCost;
          candidate.assignment = merged;
          // A repacked source's assignment entry replaces the seed's, so it
          // re-charges the ciphertext-count carrying cost at the repacked
          // placement: repacking a source fat is not free.
          bool repacked = true;
          if (repackLhs) {
            repacked &= repackPublicChain(operands[0], computeLayout,
                                          candidate.assignment, getCostModel());
          }
          if (repackRhs) {
            repacked &= repackPublicChain(operands[1], computeLayout,
                                          candidate.assignment, getCostModel());
          }
          if (!repacked) continue;
          candidate.accumulatedCost =
              accumulatedCostOf(candidate.assignment) + candidate.localCost;
          kernels.push_back(std::move(candidate));
        }
      }
      // Pack a public operand exactly like its partner (match_public_kernel
      // under the identity map). Alignment cannot reach a layout with gaps,
      // because a gap never matches a partner's traversal or replication
      // piece, so without this the consumer of a matmul result must convert
      // away from it.
      for (int side = 0; side < 2; ++side) {
        if (operands[0] == operands[1]) break;  // one value, one layout
        const Candidate& pub = side == 0 ? lhs : rhs;
        const Candidate& keep = side == 0 ? rhs : lhs;
        LayoutAttr computeLayout = keep.layout;
        if (pub.layout == computeLayout) continue;  // the aligned case above
        if (!isPublicValue(operands[side])) continue;
        if (!isLowerableRotomLayout(computeLayout)) continue;
        Candidate candidate;
        candidate.layout = computeLayout;
        candidate.kind = kind;
        candidate.operands = operandValues;
        candidate.operandLayouts = {computeLayout, computeLayout};
        candidate.hasRotomKernel =
            hasRotomKernel &&
            lowerableTriple(computeLayout, computeLayout, computeLayout);
        candidate.localCost = computeCostFn(computeLayout);
        candidate.assignment = merged;
        if (!repackPublicChain(operands[side], computeLayout,
                               candidate.assignment, getCostModel())) {
          continue;
        }
        candidate.accumulatedCost =
            accumulatedCostOf(candidate.assignment) + candidate.localCost;
        kernels.push_back(std::move(candidate));
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
      // A matmul becomes IR: explicit conversion and roll-alignment ops on
      // each operand, then a rotom.matmul op that names every layout of the
      // plan. The rewrite happens after the walk (it erases this op). Only a
      // zero-filled init qualifies; any other matmul keeps its upstream
      // lowering.
      auto fill = matmul.getOutputs()[0].getDefiningOp<linalg::FillOp>();
      if (!fill || !(matchPattern(fill.getInputs()[0], m_AnyZeroFloat()) ||
                     matchPattern(fill.getInputs()[0], m_Zero()))) {
        return;
      }
      LayoutAttr lhs = selectedLayouts.lookup(matmul.getInputs()[0]);
      LayoutAttr rhs = selectedLayouts.lookup(matmul.getInputs()[1]);
      LayoutAttr result = selectedLayouts.lookup(matmul->getResult(0));
      if (!lhs || !rhs || !result) return;
      auto shapeOf = [](Value v) {
        auto t = dyn_cast<RankedTensorType>(getPlainValueType(v.getType()));
        return t ? SmallVector<int64_t>(t.getShape().begin(),
                                        t.getShape().end())
                 : SmallVector<int64_t>();
      };
      if (std::optional<MatmulAlignment> alignment = selectMatmulAlignment(
              *this, lhs, rhs, result, shapeOf(matmul.getInputs()[0]),
              shapeOf(matmul.getInputs()[1]))) {
        matmulRewrites.push_back({matmul, *alignment});
      }
      return;
    }
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) return;
    if (!isAddLike(op) && !isMulLike(op)) return;
    auto existing = op->getAttrOfType<secret::KernelAttr>(
        secret::SecretDialect::kKernelAttrName);
    if (existing && existing.getForce()) return;

    LayoutAttr lhs = selectedLayouts.lookup(op->getOperand(0));
    LayoutAttr rhs = selectedLayouts.lookup(op->getOperand(1));
    LayoutAttr result = selectedLayouts.lookup(op->getResult(0));
    if (lowerableTriple(lhs, rhs, result)) {
      op->setAttr(kRotomElementwiseAttrName, UnitAttr::get(op->getContext()));
      // The alignment decision becomes IR: an operand at a different layout
      // is brought to the compute layout by an explicit rotom.convert_layout
      // op. The elementwise op then computes at one shared layout, and each
      // conversion lowers as its own kernel.
      OpBuilder builder(op);
      auto convertOperand = [&](unsigned index, LayoutAttr from) {
        if (from == result) return;
        auto convert = ConvertLayoutOp::create(
            builder, op->getLoc(), op->getOperand(index).getType(),
            op->getOperand(index), from, result);
        setAttributeAssociatedWith(convert.getResult(), kRotomLayoutAttrName,
                                   result);
        op->setOperand(index, convert.getResult());
      };
      convertOperand(0, lhs);
      convertOperand(1, rhs);
    } else {
      op->removeAttr(kRotomElementwiseAttrName);
    }
  });
  for (const MatmulRewrite& rewrite : matmulRewrites) {
    rewriteMatmul(rewrite);
  }
  matmulRewrites.clear();
}

// Emits the conversion ops that bring `value` from `from` to `to`, split so
// the alignment mechanism is visible: a general rotom.convert_layout to the
// unrolled placement, then one rotom.apply_roll per roll of `to`. A rolled
// source falls back to one general conversion.
Value LayoutAssignment::alignOperand(OpBuilder& builder, Location loc,
                                     Value value, ArrayRef<LayoutAttr> chain) {
  auto rollCount = [](LayoutAttr l) {
    return l.getRolls() ? l.getRolls().size() : 0;
  };
  // An apply_roll keeps the piece multiset and adds one roll. A step that
  // also moves pieces -- the sum roll splitting the summation piece, or a
  // repack mirroring the partner's pieces -- is a conversion.
  auto samePieces = [](LayoutAttr a, LayoutAttr b) {
    SmallVector<DimAttr> x = layoutDims(a), y = layoutDims(b);
    if (x.size() != y.size()) return false;
    auto less = [](DimAttr p, DimAttr q) {
      return std::tuple(p.getDim(), p.getSize(), p.getStride()) <
             std::tuple(q.getDim(), q.getSize(), q.getStride());
    };
    llvm::sort(x, less);
    llvm::sort(y, less);
    return x == y;
  };
  Value current = value;
  for (size_t i = 1; i < chain.size(); ++i) {
    LayoutAttr from = chain[i - 1], to = chain[i];
    // A step between equivalent layouts moves nothing: record the
    // layout and emit no op.
    if (from == to ||
        mergeAdjacentLayoutDims(from) == mergeAdjacentLayoutDims(to)) {
      selectedLayouts[current] = to;
      continue;
    }
    const bool roll =
        rollCount(to) == rollCount(from) + 2 && samePieces(from, to);
    Value out = roll ? ApplyRollOp::create(builder, loc, current.getType(),
                                           current, from, to)
                           .getResult()
                     : ConvertLayoutOp::create(builder, loc, current.getType(),
                                               current, from, to)
                           .getResult();
    selectedLayouts[out] = to;
    current = out;
  }
  return current;
}

void LayoutAssignment::rewriteMatmul(const MatmulRewrite& rewrite) {
  linalg::MatmulOp op = rewrite.op;
  const MatmulAlignment& alignment = rewrite.alignment;
  OpBuilder builder(op);
  Location loc = op.getLoc();
  // An operand whose chain ends in a roll by the ciphertext piece, facing a
  // public operand, keeps its pre-roll placement: the matmul folds the roll
  // into a baby-step/giant-step schedule (the reference's BSGS_ROT_ROLL +
  // BSGS_MATMUL), B - 1 + G - 1 rotations for T = B * G targets instead of T.
  auto foldableRoll = [&](ArrayRef<LayoutAttr> chain,
                          Value other) -> std::optional<BsgsSchedule> {
    if (chain.size() < 2 || !isPublicValue(other)) return std::nullopt;
    return bsgsScheduleOpt(chain[chain.size() - 2], chain.back());
  };
  auto babyExtent = [](int64_t targets) {
    int64_t b = 1;
    while (b * b < targets) ++b;
    return b;
  };
  SmallVector<LayoutAttr> lhsChain(alignment.lhsChain);
  SmallVector<LayoutAttr> rhsChain(alignment.rhsChain);
  std::optional<BsgsSchedule> fold;
  int64_t rollOperand = 0;
  if ((fold = foldableRoll(rhsChain, op.getInputs()[0]))) {
    rollOperand = 1;
    rhsChain.pop_back();
  } else if ((fold = foldableRoll(lhsChain, op.getInputs()[1]))) {
    rollOperand = 0;
    lhsChain.pop_back();
  }
  Value lhs = alignOperand(builder, loc, op.getInputs()[0], lhsChain);
  Value rhs = alignOperand(builder, loc, op.getInputs()[1], rhsChain);
  Value result;
  if (fold) {
    // The folded operand stops one step short of its chain: the kernel reads
    // it unrolled and rotates each target itself.
    LayoutAttr rolled = rollOperand == 1 ? alignment.rhsChain.back()
                                         : alignment.lhsChain.back();
    result = BsgsMatmulOp::create(builder, loc, op.getResult(0).getType(), lhs,
                                  rhs, lhsChain.back(), rhsChain.back(), rolled,
                                  alignment.compute, alignment.result,
                                  rollOperand, fold->stride, fold->targets,
                                  babyExtent(fold->targets), fold->negative)
                 .getResult();
  } else {
    result =
        MatmulOp::create(builder, loc, op.getResult(0).getType(), lhs, rhs,
                         alignment.lhsChain.back(), alignment.rhsChain.back(),
                         alignment.compute, alignment.result)
            .getResult();
  }
  selectedLayouts.erase(op->getResult(0));
  selectedLayouts[result] = alignment.result;
  op.getResult(0).replaceAllUsesWith(result);
  Value init = op.getOutputs()[0];
  op.erase();
  // The zero-init chain fed only the erased matmul: erase what died with it
  // (the fill, its zero constant, and its tensor.empty), and drop the dead
  // values from the selection map.
  SmallVector<Value> worklist = {init};
  while (!worklist.empty()) {
    Operation* def = worklist.pop_back_val().getDefiningOp();
    if (!def || !def->use_empty()) continue;
    worklist.append(def->operand_begin(), def->operand_end());
    for (Value result : def->getResults()) selectedLayouts.erase(result);
    def->erase();
  }
}

void LayoutAssignment::runOnOperation() {
  splitConstantsPerUse(getOperation());

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
