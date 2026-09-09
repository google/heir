// The two-operand generators: elementwise (add/sub/mul and the elementwise
// linalg.generic) and matmul. Both run on the shared alignment engine of
// LayoutAlignment -- the reference's gen_binop -- and differ only in the
// operator map (identity vs. matmul), in whether the sum roll is offered,
// and in how the result layout is derived.

#include <cstdint>
#include <optional>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/AssignmentContext.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/CostModel.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/DimMaps.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Generators.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/ValueUtils.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Dialect/Rotom/Utils/LayoutConversion.h"
#include "lib/Dialect/Rotom/Utils/RotomLayout.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::rotom {

// Defined by the layout assignment pass: a value whose data is known at
// encode time, so a matmul may fold a partner's roll into a BSGS schedule.
bool isPublicValue(Value value);

// What the rewrite will really charge for a chain that ends in a roll by the
// ciphertext piece facing a public operand: the fold replaces one rotation
// per target with B - 1 baby and G - 1 giant rotations. Returns the signed
// adjustment to the chain's price, or 0 when no fold applies.
static int64_t bsgsAdjustment(AssignmentContext& ctx,
                              ArrayRef<LayoutAttr> chain, Value other) {
  if (chain.size() < 2 || !isPublicValue(other)) return 0;
  LayoutAttr from = chain[chain.size() - 2], to = chain.back();
  std::optional<BsgsSchedule> roll = bsgsScheduleOpt(from, to);
  if (!roll) return 0;
  int64_t baby = 1;
  while (baby * baby < roll->targets) ++baby;
  const int64_t giants = (roll->targets + baby - 1) / baby;
  const RotomCostModel& costModel = getCostModel();
  // The step as priced now, against what the fold emits.
  const int64_t asPriced = ctx.conversionCost(from, to);
  const int64_t folded = ((baby - 1) + (giants - 1)) * costModel.rotation;
  return folded - asPriced;
}

// ---- elementwise ----------------------------------------------------------

// A linalg.generic is elementwise when every operand is read with the identity
// indexing map under purely parallel iteration -- no broadcast, reduction, or
// permutation -- so a layout can pass through it unchanged.
static bool isElementwiseGeneric(linalg::GenericOp op) {
  for (AffineMap map : op.getIndexingMapsArray()) {
    if (!map.isIdentity()) return false;
  }
  for (utils::IteratorType iteratorType : op.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) return false;
  }
  return true;
}

// True when the body is a single add-like op (ignoring the yield and any
// constants), i.e. the generic computes an elementwise addition.
static bool hasAddLikeBody(linalg::GenericOp op) {
  bool foundAddLikeOp = false;
  for (Operation& innerOp : op.getBody()->getOperations()) {
    if (isa<linalg::YieldOp, arith::ConstantOp>(innerOp)) continue;
    if (!isAddLike(&innerOp)) return false;
    foundAddLikeOp = true;
  }
  return foundAddLikeOp;
}

LogicalResult generateElementwise(AssignmentContext& ctx, Operation* op) {
  if (op->getNumOperands() == 2) {
    const bool hasRotomKernel = isAddLike(op) || isMulLike(op);
    SmallVector<Value> operands = {op->getOperand(0), op->getOperand(1)};
    SmallVector<Candidate> kernels = ctx.chooseElementwiseKernels(
        operands, KernelKind::Elementwise,
        [&](LayoutAttr layout) { return operationCost(op, layout); },
        hasRotomKernel);
    if (!kernels.empty()) {
      ctx.assignResultsFromCandidates(op, kernels);
      return success();
    }
  }

  SmallVector<Candidate> chosen =
      ctx.chooseCommonOperandCandidates(op, KernelKind::Elementwise);
  ctx.assignResultsFromCandidates(op, chosen);
  return success();
}

LogicalResult generateLinalgGeneric(AssignmentContext& ctx,
                                    linalg::GenericOp op) {
  if (!isElementwiseGeneric(op)) return generatePassThrough(ctx, op);
  if (hasAddLikeBody(op) && op.getInputs().size() == 2) {
    SmallVector<Value> operands = {op.getInputs()[0], op.getInputs()[1]};
    SmallVector<Candidate> kernels = ctx.chooseElementwiseKernels(
        operands, KernelKind::Generic,
        [&](LayoutAttr layout) { return genericOperationCost(op, layout); });
    if (!kernels.empty()) {
      ctx.assignResultsFromCandidates(op, kernels);
      return success();
    }
  }

  SmallVector<Value> operands;
  SmallVector<SmallVector<Candidate>> candidateSets;
  for (Value operand : op->getOperands()) {
    if (!isTensorLike(operand)) continue;
    SmallVector<Candidate> operandCandidates = ctx.candidatesForValue(operand);
    if (operandCandidates.empty()) continue;
    operands.push_back(operand);
    candidateSets.push_back(operandCandidates);
  }
  SmallVector<Candidate> chosen = chooseCommonCandidates(
      operands, candidateSets, KernelKind::Generic,
      [&](LayoutAttr layout) { return genericOperationCost(op, layout); },
      [&ctx](LayoutAttr from, LayoutAttr to) {
        return ctx.conversionCost(from, to);
      });
  ctx.assignResultsFromCandidates(op, chosen);
  return success();
}

// ---- matmul ---------------------------------------------------------------

// The operands' axes read as the matmul iteration space, for any ranks: the
// lhs is [..batch, i, k] and the rhs is [..batch, k, j], with a rank-1 operand
// standing for [k]. A non-contraction axis takes its id in the result
// [..batch, i, j]; the contraction takes kDimContraction, since the result
// sums it out and carries a gap there instead.
struct MatmulAxes {
  int64_t lhsRank = 2, rhsRank = 2;
  int64_t lhsSumDim() const { return lhsRank - 1; }
  int64_t rhsSumDim() const { return rhsRank == 1 ? 0 : rhsRank - 2; }
  int64_t batch() const {
    return std::max(std::max<int64_t>(lhsRank - 2, 0),
                    std::max<int64_t>(rhsRank - 2, 0));
  }
  int64_t lhsIter(int64_t axis) const {
    if (axis == lhsSumDim()) return kDimContraction;
    if (axis == lhsRank - 2) return batch();
    return batch() - (lhsRank - 2) + axis;
  }
  int64_t rhsIter(int64_t axis) const {
    if (axis == rhsSumDim()) return kDimContraction;
    if (axis == rhsRank - 1) return batch() + 1;
    return batch() - (rhsRank - 2) + axis;
  }
};
namespace {
// One priced way to run the matmul: the per-operand chains from the
// operand's layout to its aligned placement, the compute placement, and the
// result layout.
struct MatmulOption {
  SmallVector<LayoutAttr> lhsChain;
  SmallVector<LayoutAttr> rhsChain;
  LayoutAttr compute;
  LayoutAttr result;
};
}  // namespace

// The placement the elementwise product computes at, over the (i, j, k)
// iteration space: piece by piece from the aligned pair, a traversal piece
// naming its iteration axis and two replications staying replication.
static LayoutAttr computeLayoutOf(const MatmulAxes& axes, LayoutAttr lhs,
                                  LayoutAttr rhs) {
  MLIRContext* ctx = lhs.getContext();
  SmallVector<DimAttr> a = layoutDims(lhs), b = layoutDims(rhs);
  SmallVector<DimAttr> out;
  auto trav = [](DimAttr d) { return !d.isGap() && !d.isReplicate(); };
  // The pair arrives aligned, so the two piece lists already correspond one
  // to one and the walk steps them together. Alignment restates both sides at
  // one granularity, and canonicalization pins a unit piece to the tail of
  // each list, so a unit piece faces its partner rather than a data piece.
  size_t ia = 0, ib = 0;
  while (ia < a.size() || ib < b.size()) {
    if (ia >= a.size() || ib >= b.size()) return {};
    DimAttr x = a[ia++], y = b[ib++];
    if (trav(x)) {
      out.push_back(DimAttr::get(ctx, axes.lhsIter(x.getDim()), x.getSize(),
                                 x.getStride()));
    } else if (trav(y)) {
      out.push_back(DimAttr::get(ctx, axes.rhsIter(y.getDim()), y.getSize(),
                                 y.getStride()));
    } else {
      out.push_back(x);  // both replication, or a shared gap
    }
  }
  LayoutAttr compute = LayoutAttr::getCanonical(ctx, out, lhs.getN());
  return compute ? mergeAdjacentLayoutDims(compute) : LayoutAttr();
}

// The compute-and-reduce price of one option, exclusive of alignment: one
// multiply per compute ciphertext, then the k reduction -- a log tree of
// rotate-and-add for each slot-region k piece, and block adds for the
// ciphertext-region k pieces.
static int64_t matmulKernelCost(LayoutAttr compute) {
  const RotomCostModel& costModel = getCostModel();
  SmallVector<DimAttr> dims = layoutDims(compute);
  const size_t ctLen = inferCtPrefixLen(dims, compute.getN());
  const int64_t numCtCompute = layoutNumCiphertexts(compute);
  int64_t ctK = 1, reduceSteps = 0;
  for (size_t p = 0; p < dims.size(); ++p) {
    DimAttr d = dims[p];
    if (!d.isContraction()) continue;
    if (p < ctLen) {
      ctK *= d.getSize();
    } else {
      reduceSteps += llvm::Log2_64_Ceil(d.getSize());
    }
  }
  const int64_t numCtResult = numCtCompute / ctK;
  const int64_t reduceRotations = numCtResult * reduceSteps;
  const int64_t reduceAdds =
      numCtResult * reduceSteps + (numCtCompute - numCtResult);
  return numCtCompute * costModel.ciphertextMultiply +
         reduceRotations * costModel.rotation + reduceAdds * costModel.add;
}

// Price of a chain: one estimate per step (conversions and rolls alike).
// Nullopt when a step's target cannot be lowered.
static std::optional<int64_t> chainCost(AssignmentContext& ctx,
                                        ArrayRef<LayoutAttr> chain) {
  int64_t cost = 0;
  for (size_t i = 1; i < chain.size(); ++i) {
    if (!isLowerableRotomLayout(chain[i])) return std::nullopt;
    const int64_t step = ctx.conversionCost(chain[i - 1], chain[i]);
    // A step the lowering cannot emit makes the whole chain unusable.
    if (step >= AssignmentContext::kUnlowerableConversion) return std::nullopt;
    cost += step;
  }
  return cost;
}

static void extend(SmallVector<LayoutAttr>& chain, LayoutAttr next) {
  if (chain.empty() || chain.back() != next) chain.push_back(next);
}

// The reference's gen_binop for matmul: replicate, the sum-roll variants
// (neither, either, both sides), align each pair, derive the result.
static SmallVector<MatmulOption> enumerateMatmulOptions(
    LayoutAttr lhs, LayoutAttr rhs, ArrayRef<int64_t> lhsShape,
    ArrayRef<int64_t> rhsShape) {
  SmallVector<MatmulOption> options;
  const MatmulAxes axes{static_cast<int64_t>(lhsShape.size()),
                        static_cast<int64_t>(rhsShape.size())};
  auto map = OperatorAlignmentMap::matmul(axes.lhsRank, axes.rhsRank);
  auto rep = replicateForAlignment(map, lhs, rhs, lhsShape, rhsShape);

  if (failed(rep)) return options;

  struct Pair {
    SmallVector<LayoutAttr> lhsChain, rhsChain;
  };
  SmallVector<Pair> pairs;
  Pair base;
  extend(base.lhsChain, lhs);
  extend(base.lhsChain, rep->lhs);
  extend(base.rhsChain, rhs);
  extend(base.rhsChain, rep->rhs);
  pairs.push_back(base);
  std::optional<LayoutAttr> rolledLhs =
      applySumRoll(rep->lhs, axes.lhsSumDim());
  std::optional<LayoutAttr> rolledRhs =
      applySumRoll(rep->rhs, axes.rhsSumDim());
  if (rolledLhs && rolledRhs) {
    Pair p = base;
    extend(p.lhsChain, *rolledLhs);
    extend(p.rhsChain, *rolledRhs);
    pairs.push_back(p);
  }
  if (rolledRhs) {
    Pair p = base;
    extend(p.rhsChain, *rolledRhs);
    pairs.push_back(p);
  }
  if (rolledLhs) {
    Pair p = base;
    extend(p.lhsChain, *rolledLhs);
    pairs.push_back(p);
  }

  for (const Pair& pair : pairs) {
    auto dbgPairs = alignPair(map, pair.lhsChain.back(), pair.rhsChain.back());

    for (const AlignedPair& aligned : dbgPairs) {
      MatmulOption option;
      option.lhsChain = pair.lhsChain;
      option.rhsChain = pair.rhsChain;
      extend(option.lhsChain, aligned.lhs);
      extend(option.rhsChain, aligned.rhs);
      option.compute = computeLayoutOf(axes, aligned.lhs, aligned.rhs);
      std::optional<LayoutAttr> result =
          outputLayout(map, /*isMatmul=*/true, aligned.lhs, aligned.rhs,
                       axes.lhsSumDim(), axes.rhsSumDim());

      if (!option.compute || !result) continue;
      option.result = *result;
      options.push_back(std::move(option));
    }
  }
  return options;
}

// A source candidate is data packed at encode time rather than computed
// homomorphically (the reference's TENSOR kernel, which match_public_kernel
// packs to whatever the other side needs). Its aligned placement is
// reachable for free by assigning it as the source's own layout.
static bool isSourceCandidate(const Candidate& candidate) {
  return candidate.kind == KernelKind::Tensor;
}

// The near-zero price of repacking a source at a demanded placement: enough
// to lose ties against an option whose operand already sits at its seeded
// layout, far below any ciphertext conversion.
constexpr int64_t kSourceRepackCost = 1;

namespace {
struct AlignChoice {
  LayoutAttr layout;  // the operand's assigned layout under this choice
  SmallVector<LayoutAttr> chain;
  int64_t alignCost;
  bool repack;
};
}  // namespace

static SmallVector<AlignChoice, 2> alignChoices(AssignmentContext& ctx,
                                                const Candidate& operand,
                                                ArrayRef<LayoutAttr> chain) {
  SmallVector<AlignChoice, 2> choices;
  if (std::optional<int64_t> cost = chainCost(ctx, chain)) {
    choices.push_back({operand.layout, SmallVector<LayoutAttr>(chain), *cost,
                       /*repack=*/false});
  }
  LayoutAttr aligned = chain.back();
  if (isSourceCandidate(operand) && operand.layout != aligned &&
      isLowerableRotomLayout(aligned)) {
    choices.push_back({aligned, {aligned}, 0, /*repack=*/true});
  }
  return choices;
}

static SmallVector<int64_t> shapeOf(Value value) {
  auto type = dyn_cast<RankedTensorType>(getPlainValueType(value.getType()));
  if (!type) return {};
  return SmallVector<int64_t>(type.getShape().begin(), type.getShape().end());
}

std::optional<MatmulAlignment> selectMatmulAlignment(
    AssignmentContext& ctx, LayoutAttr lhs, LayoutAttr rhs, LayoutAttr result,
    ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape) {
  std::optional<MatmulAlignment> best;
  std::optional<int64_t> bestCost;
  for (const MatmulOption& option :
       enumerateMatmulOptions(lhs, rhs, lhsShape, rhsShape)) {
    if (option.result != result) continue;
    std::optional<int64_t> lhsCost = chainCost(ctx, option.lhsChain);
    std::optional<int64_t> rhsCost = chainCost(ctx, option.rhsChain);
    if (!lhsCost || !rhsCost) continue;
    const int64_t cost = *lhsCost + *rhsCost + matmulKernelCost(option.compute);
    if (!bestCost || cost < *bestCost) {
      best = MatmulAlignment{option.lhsChain, option.rhsChain, option.compute,
                             option.result};
      bestCost = cost;
    }
  }
  return best;
}

LogicalResult generateMatmul(AssignmentContext& ctx, linalg::MatmulOp op) {
  // Only the default (i,k) x (k,j) indexing; transposed/broadcast variants
  // pass through until they get their own dim maps.
  if (op.hasUserDefinedMaps()) return generatePassThrough(ctx, op);

  Value lhs = op.getInputs()[0];
  Value rhs = op.getInputs()[1];
  SmallVector<int64_t> lhsShape = shapeOf(lhs), rhsShape = shapeOf(rhs);
  SmallVector<Candidate> lhsCandidates = ctx.candidatesForValue(lhs);
  SmallVector<Candidate> rhsCandidates = ctx.candidatesForValue(rhs);
  const RotomCostModel& costModel = getCostModel();

  SmallVector<Candidate> candidates;
  for (const Candidate& lhsCandidate : lhsCandidates) {
    for (const Candidate& rhsCandidate : rhsCandidates) {
      Assignment merged;
      if (!mergeAssignments(merged, lhsCandidate.assignment) ||
          !mergeAssignments(merged, rhsCandidate.assignment)) {
        continue;
      }
      SmallVector<MatmulOption> dbgOptions = enumerateMatmulOptions(
          lhsCandidate.layout, rhsCandidate.layout, lhsShape, rhsShape);

      for (const MatmulOption& option : dbgOptions) {
        const int64_t kernelCost = matmulKernelCost(option.compute);

        for (const AlignChoice& lhsChoice :
             alignChoices(ctx, lhsCandidate, option.lhsChain)) {
          for (const AlignChoice& rhsChoice :
               alignChoices(ctx, rhsCandidate, option.rhsChain)) {
            // A self-matmul shares one source: both sides must agree on its
            // single assigned layout.
            if (lhs == rhs && (lhsChoice.repack != rhsChoice.repack ||
                               lhsChoice.layout != rhsChoice.layout)) {
              continue;
            }
            Candidate candidate;
            candidate.layout = option.result;
            candidate.kind = KernelKind::Matmul;
            candidate.operands = {lhs, rhs};
            candidate.operandLayouts = {lhsChoice.layout, rhsChoice.layout};
            candidate.localCost =
                lhsChoice.alignCost + rhsChoice.alignCost + kernelCost;
            // Only a chain the rewrite will actually keep can be folded: a
            // repacked source arrives already placed, with no chain to fold.
            if (!rhsChoice.repack) {
              candidate.localCost += bsgsAdjustment(ctx, rhsChoice.chain, lhs);
            }
            if (!lhsChoice.repack) {
              candidate.localCost += bsgsAdjustment(ctx, lhsChoice.chain, rhs);
            }
            candidate.assignment = merged;
            // A repacked source's assignment entry replaces the seed's, so
            // it re-charges the ciphertext-count carrying cost at the
            // repacked placement: repacking a source fat is not free.
            if (lhsChoice.repack) {
              candidate.assignment[lhs] = {
                  lhsChoice.layout,
                  kSourceRepackCost +
                      costModel.ciphertextCount *
                          layoutNumCiphertexts(lhsChoice.layout)};
            }
            if (rhsChoice.repack) {
              candidate.assignment[rhs] = {
                  rhsChoice.layout,
                  kSourceRepackCost +
                      costModel.ciphertextCount *
                          layoutNumCiphertexts(rhsChoice.layout)};
            }
            candidate.accumulatedCost =
                accumulatedCostOf(candidate.assignment) + candidate.localCost;

            candidates.push_back(std::move(candidate));
          }
        }
      }
    }
  }
  ctx.assignResultsFromCandidates(op, uniqueCandidates(candidates));
  return success();
}

}  // namespace mlir::heir::rotom
