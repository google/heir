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
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/CostModel.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/DimMaps.h"
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
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"     // from @llvm-project

namespace mlir::heir::rotom {

namespace {

constexpr llvm::StringLiteral kRotomSeedAttrName = "rotom.seed";
constexpr llvm::StringLiteral kRotomLayoutAttrName = "rotom.layout";

}  // namespace

#define DEBUG_TYPE "rotom-assign-layout"

#define GEN_PASS_DEF_LAYOUTASSIGNMENT
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/LayoutAssignment.h.inc"

namespace {

Type getPlainValueType(Type type) {
  if (auto secretType = dyn_cast<secret::SecretType>(type)) {
    return secretType.getValueType();
  }
  return type;
}

bool isTensorLike(Value value) {
  return isa<RankedTensorType>(getPlainValueType(value.getType()));
}

bool isLayoutCompatibleWithValue(LayoutAttr layout, Value value) {
  auto type = dyn_cast<RankedTensorType>(getPlainValueType(value.getType()));
  if (!type) return false;

  int64_t rank = type.getRank();
  for (Attribute attr : layout.getDims()) {
    auto dim = cast<DimAttr>(attr);
    if (dim.isGap() || dim.isReplicate()) continue;
    int64_t dimIndex = dim.getDim();
    if (dimIndex >= rank) return false;
    int64_t typeDimSize = type.getDimSize(dimIndex);
    if (typeDimSize == ShapedType::kDynamic) continue;
    if (typeDimSize <= 0) continue;
    int64_t paddedDimSize = nextPowerOfTwo(typeDimSize);
    if (dim.getSize() * dim.getStride() > paddedDimSize) return false;
  }
  return true;
}

bool isElementwiseGeneric(linalg::GenericOp op) {
  for (AffineMap map : op.getIndexingMapsArray()) {
    if (!map.isIdentity()) return false;
  }
  for (utils::IteratorType iteratorType : op.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) return false;
  }
  return true;
}

bool hasAddLikeBody(linalg::GenericOp op) {
  bool foundAddLikeOp = false;
  for (Operation& innerOp : op.getBody()->getOperations()) {
    if (isa<linalg::YieldOp, arith::ConstantOp>(innerOp)) continue;
    if (!isAddLike(&innerOp)) return false;
    foundAddLikeOp = true;
  }
  return foundAddLikeOp;
}

struct LayoutAssignment : public impl::LayoutAssignmentBase<LayoutAssignment> {
  using LayoutAssignmentBase::LayoutAssignmentBase;

  DenseMap<Value, SmallVector<Candidate>> candidates;
  DenseMap<Value, LayoutAttr> selectedLayouts;
  DenseMap<std::pair<Attribute, Attribute>, int64_t> conversionCostCache;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<secret::SecretDialect>();
    registry.insert<tensor_ext::TensorExtDialect>();
  }

  void runOnOperation() override;

  // --- Candidate generation: forward pass that fills `candidates` for every
  // value. Each tensor op has its own visitor (visit*); these are kept
  // self-contained -- depending only on the shared seed/candidate/cost helpers
  // -- so that as the per-op kernel space grows they can move to one file each.
  LogicalResult generateCandidates(ModuleOp module);
  LogicalResult visitOperation(Operation* op);
  void seedValue(Value value);
  void setCandidates(Value value, ArrayRef<Candidate> newCandidates);
  SmallVector<Candidate> candidatesForValue(Value value);
  void assignResultsFromCandidates(Operation* op, ArrayRef<Candidate> chosen);
  SmallVector<Candidate> chooseCommonOperandCandidates(Operation* op);
  SmallVector<Candidate> chooseCommonOperandCandidates(Operation* op,
                                                       KernelKind kind);
  SmallVector<Candidate> chooseElementwiseKernels(
      ArrayRef<Value> operands, KernelKind kind,
      function_ref<int64_t(LayoutAttr)> computeCostFn,
      std::optional<KernelName> rotomKernel = std::nullopt);
  int64_t cachedConversionCost(LayoutAttr from, LayoutAttr to);
  LogicalResult visitFunc(func::FuncOp op);
  LogicalResult visitGeneric(secret::GenericOp op);
  LogicalResult visitYield(secret::YieldOp op);
  LogicalResult visitPassThrough(Operation* op);
  LogicalResult visitElementwise(Operation* op);
  LogicalResult visitGeneric(linalg::GenericOp op);
  LogicalResult visitTranspose(linalg::TransposeOp op);
  LogicalResult visitReduction(linalg::ReduceOp op);
  LogicalResult visitCollapseShape(tensor::CollapseShapeOp op);
  LogicalResult visitExpandShape(tensor::ExpandShapeOp op);
  LogicalResult visitExtractSlice(tensor::ExtractSliceOp op);
  LogicalResult visitInsertSlice(tensor::InsertSliceOp op);

  // --- Layout search: each value's candidates already carry the dedup'd
  // accumulated cost and the full assignment of everything feeding it, so
  // selection is just picking the cheapest consistent assignment at each
  // function's returned values -- no backward propagation. ---
  void selectLayouts(ModuleOp module);
  LogicalResult visitReturn(func::ReturnOp op);
  void applyKernels(ModuleOp module);
  void writeSelectedLayouts();
};

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
    Operation* op) {
  return chooseCommonOperandCandidates(op, KernelKind::PassThrough);
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

  // Real Vos-Vos-Erkin rotation count weighted by the per-rotation cost. Every
  // candidate layout is materializable (the search only generates such
  // layouts), so the shift network always yields a cost. Cached per layout pair
  // since the search queries the same pairs across many candidate pairings.
  std::optional<int64_t> rotations = shiftNetworkConversionCost(from, to);
  assert(rotations && "shift network must yield a cost for assignable layouts");
  int64_t cost = *rotations * getCostModel().rotation;
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

LogicalResult LayoutAssignment::visitFunc(func::FuncOp op) {
  for (Value arg : op.getArguments()) seedValue(arg);
  return success();
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

LogicalResult LayoutAssignment::visitGeneric(secret::GenericOp op) {
  for (OpOperand& operand : op->getOpOperands()) {
    SmallVector<Candidate> operandCandidates =
        candidatesForValue(operand.get());
    if (operandCandidates.empty()) continue;
    BlockArgument blockArg =
        op.getRegion().getArgument(operand.getOperandNumber());
    SmallVector<Candidate> blockArgCandidates;
    for (const Candidate& candidate : operandCandidates) {
      // The block argument is the operand routed into the region: same data,
      // same assignment, no extra cost.
      Candidate blockArgCandidate;
      blockArgCandidate.layout = candidate.layout;
      blockArgCandidate.kind = KernelKind::BlockArgument;
      blockArgCandidate.operands = {operand.get()};
      blockArgCandidate.operandLayouts = {candidate.layout};
      blockArgCandidate.assignment = candidate.assignment;
      blockArgCandidate.accumulatedCost =
          accumulatedCostOf(blockArgCandidate.assignment);
      blockArgCandidates.push_back(std::move(blockArgCandidate));
    }
    setCandidates(blockArg, blockArgCandidates);
  }
  return success();
}

LogicalResult LayoutAssignment::visitYield(secret::YieldOp op) {
  auto generic = op->getParentOfType<secret::GenericOp>();
  for (OpOperand& operand : op->getOpOperands()) {
    SmallVector<Candidate> yielded = candidatesForValue(operand.get());
    if (yielded.empty()) continue;
    SmallVector<Candidate> resultCandidates;
    for (const Candidate& candidate : yielded) {
      // The generic's result is the yielded value: same data, same assignment.
      Candidate resultCandidate;
      resultCandidate.layout = candidate.layout;
      resultCandidate.kind = KernelKind::Yield;
      resultCandidate.operands = {operand.get()};
      resultCandidate.operandLayouts = {candidate.layout};
      resultCandidate.assignment = candidate.assignment;
      resultCandidate.accumulatedCost =
          accumulatedCostOf(resultCandidate.assignment);
      resultCandidates.push_back(std::move(resultCandidate));
    }
    setCandidates(generic.getResult(operand.getOperandNumber()),
                  resultCandidates);
  }
  return success();
}

LogicalResult LayoutAssignment::visitPassThrough(Operation* op) {
  SmallVector<Candidate> chosen = chooseCommonOperandCandidates(op);
  if (chosen.empty()) {
    for (Value result : op->getResults()) seedValue(result);
    return success();
  }
  assignResultsFromCandidates(op, chosen);
  return success();
}

LogicalResult LayoutAssignment::visitElementwise(Operation* op) {
  if (op->getNumOperands() == 2) {
    std::optional<KernelName> rotomKernel = selectRotomElementwiseKernel(op);
    SmallVector<Value> operands = {op->getOperand(0), op->getOperand(1)};
    SmallVector<Candidate> kernels = chooseElementwiseKernels(
        operands, KernelKind::Elementwise,
        [&](LayoutAttr layout) { return operationCost(op, layout); },
        rotomKernel);
    if (!kernels.empty()) {
      assignResultsFromCandidates(op, kernels);
      return success();
    }
  }

  SmallVector<Candidate> chosen =
      chooseCommonOperandCandidates(op, KernelKind::Elementwise);
  assignResultsFromCandidates(op, chosen);
  return success();
}

LogicalResult LayoutAssignment::visitGeneric(linalg::GenericOp op) {
  if (!isElementwiseGeneric(op)) return visitPassThrough(op);
  if (hasAddLikeBody(op) && op.getInputs().size() == 2) {
    SmallVector<Value> operands = {op.getInputs()[0], op.getInputs()[1]};
    SmallVector<Candidate> kernels = chooseElementwiseKernels(
        operands, KernelKind::Generic,
        [&](LayoutAttr layout) { return genericOperationCost(op, layout); });
    if (!kernels.empty()) {
      assignResultsFromCandidates(op, kernels);
      return success();
    }
  }

  SmallVector<Value> operands;
  SmallVector<SmallVector<Candidate>> candidateSets;
  for (Value operand : op->getOperands()) {
    if (!isTensorLike(operand)) continue;
    SmallVector<Candidate> operandCandidates = candidatesForValue(operand);
    if (operandCandidates.empty()) continue;
    operands.push_back(operand);
    candidateSets.push_back(operandCandidates);
  }
  SmallVector<Candidate> chosen = chooseCommonCandidates(
      operands, candidateSets, KernelKind::Generic,
      [&](LayoutAttr layout) { return genericOperationCost(op, layout); },
      [this](LayoutAttr from, LayoutAttr to) {
        return cachedConversionCost(from, to);
      });
  assignResultsFromCandidates(op, chosen);
  return success();
}

LogicalResult LayoutAssignment::visitTranspose(linalg::TransposeOp op) {
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!inputType) return visitPassThrough(op);

  SmallVector<int64_t> oldToNew(inputType.getRank(), -2);
  for (auto [outputDim, inputDim] : llvm::enumerate(op.getPermutation())) {
    if (inputDim < 0 || inputDim >= inputType.getRank()) {
      return visitPassThrough(op);
    }
    oldToNew[inputDim] = static_cast<int64_t>(outputDim);
  }

  SmallVector<Candidate> inputCandidates = candidatesForValue(op.getInput());
  SmallVector<Candidate> transposed = remapCandidates(
      op.getInput(), inputCandidates, oldToNew, KernelKind::Transpose);
  assignResultsFromCandidates(op, transposed);
  return success();
}

LogicalResult LayoutAssignment::visitReduction(linalg::ReduceOp op) {
  for (auto [input, result] : llvm::zip(op.getInputs(), op.getResults())) {
    SmallVector<Candidate> inputCandidates = candidatesForValue(input);
    if (inputCandidates.empty()) continue;

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) continue;

    std::optional<SmallVector<int64_t>> oldToNew =
        getReductionDimMap(inputType.getRank(), op.getDimensions());
    if (!oldToNew) continue;

    SmallVector<Candidate> reduced =
        remapCandidates(input, inputCandidates, *oldToNew, KernelKind::Reduce);
    for (Candidate& candidate : reduced) {
      // A reduction sums its input ciphertexts, so its local cost is set by the
      // aligned INPUT layout (operandLayouts[0]) -- one add per input
      // ciphertext -- not the smaller reduced output layout.
      if (candidate.operandLayouts.empty()) continue;
      int64_t inputNumCt = layoutNumCiphertexts(candidate.operandLayouts[0]);
      candidate.localCost += inputNumCt * getCostModel().add;
    }
    setCandidates(result, reduced);
  }
  return success();
}

LogicalResult LayoutAssignment::visitCollapseShape(tensor::CollapseShapeOp op) {
  std::optional<SmallVector<int64_t>> oldToNew =
      getCollapseShapeDimMap(op.getSrcType(), op.getReassociationIndices());
  if (!oldToNew) return visitPassThrough(op);

  SmallVector<Candidate> collapsed =
      remapCandidates(op.getSrc(), candidatesForValue(op.getSrc()), *oldToNew,
                      KernelKind::CollapseShape);
  assignResultsFromCandidates(op, collapsed);
  return success();
}

LogicalResult LayoutAssignment::visitExpandShape(tensor::ExpandShapeOp op) {
  std::optional<SmallVector<int64_t>> oldToNew =
      getExpandShapeDimMap(op.getResultType(), op.getReassociationIndices());
  if (!oldToNew) return visitPassThrough(op);

  SmallVector<Candidate> expanded =
      remapCandidates(op.getSrc(), candidatesForValue(op.getSrc()), *oldToNew,
                      KernelKind::ExpandShape);
  assignResultsFromCandidates(op, expanded);
  return success();
}

LogicalResult LayoutAssignment::visitExtractSlice(tensor::ExtractSliceOp op) {
  std::optional<SmallVector<int64_t>> oldToNew = getExtractSliceDimMap(
      op.getResultType(), op.getStaticSizes(), op.getStaticStrides());
  if (!oldToNew) return visitPassThrough(op);

  SmallVector<Candidate> sliced =
      remapCandidates(op.getSource(), candidatesForValue(op.getSource()),
                      *oldToNew, KernelKind::ExtractSlice);
  assignResultsFromCandidates(op, sliced);
  return success();
}

LogicalResult LayoutAssignment::visitInsertSlice(tensor::InsertSliceOp op) {
  SmallVector<Candidate> destCandidates = candidatesForValue(op.getDest());
  if (!destCandidates.empty()) {
    SmallVector<Candidate> sourceCandidates =
        candidatesForValue(op.getSource());
    std::optional<SmallVector<int64_t>> sourceToDest =
        getInsertSliceDimMap(op.getSourceType(), op.getResultType(),
                             op.getStaticSizes(), op.getStaticStrides());
    if (sourceToDest) {
      SmallVector<Candidate> expandedSource =
          remapCandidates(op.getSource(), sourceCandidates, *sourceToDest,
                          KernelKind::InsertSlice);
      if (!expandedSource.empty()) {
        SmallVector<Value> operands = {op.getDest(), op.getSource()};
        SmallVector<SmallVector<Candidate>> sets = {destCandidates,
                                                    expandedSource};
        assignResultsFromCandidates(op,
                                    chooseCommonCandidates(
                                        operands, sets, KernelKind::InsertSlice,
                                        [](LayoutAttr) { return 0; },
                                        [this](LayoutAttr from, LayoutAttr to) {
                                          return cachedConversionCost(from, to);
                                        }));
        return success();
      }
    }
    assignResultsFromCandidates(op, destCandidates);
    return success();
  }

  std::optional<SmallVector<int64_t>> sourceToDest =
      getInsertSliceDimMap(op.getSourceType(), op.getResultType(),
                           op.getStaticSizes(), op.getStaticStrides());
  if (!sourceToDest) return visitPassThrough(op);

  SmallVector<Candidate> expandedSource =
      remapCandidates(op.getSource(), candidatesForValue(op.getSource()),
                      *sourceToDest, KernelKind::InsertSlice, /*extraCost=*/1);
  assignResultsFromCandidates(op, expandedSource);
  return success();
}

LogicalResult LayoutAssignment::visitOperation(Operation* op) {
  return TypeSwitch<Operation*, LogicalResult>(op)
      .Case<func::FuncOp>([&](auto typedOp) { return visitFunc(typedOp); })
      .Case<secret::GenericOp>(
          [&](auto typedOp) { return visitGeneric(typedOp); })
      .Case<secret::YieldOp>([&](auto typedOp) { return visitYield(typedOp); })
      .Case<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
            arith::MulFOp, arith::MulIOp>(
          [&](auto typedOp) { return visitElementwise(typedOp); })
      .Case<linalg::GenericOp>(
          [&](auto typedOp) { return visitGeneric(typedOp); })
      .Case<linalg::TransposeOp>(
          [&](auto typedOp) { return visitTranspose(typedOp); })
      .Case<linalg::ReduceOp>(
          [&](auto typedOp) { return visitReduction(typedOp); })
      .Case<tensor::CollapseShapeOp>(
          [&](auto typedOp) { return visitCollapseShape(typedOp); })
      .Case<tensor::ExpandShapeOp>(
          [&](auto typedOp) { return visitExpandShape(typedOp); })
      .Case<tensor::ExtractSliceOp>(
          [&](auto typedOp) { return visitExtractSlice(typedOp); })
      .Case<tensor::InsertSliceOp>(
          [&](auto typedOp) { return visitInsertSlice(typedOp); })
      .Default(
          [&](Operation* genericOp) { return visitPassThrough(genericOp); });
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

}  // namespace

}  // namespace mlir::heir::rotom
