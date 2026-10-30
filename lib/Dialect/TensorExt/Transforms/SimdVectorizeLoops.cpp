#include "lib/Dialect/TensorExt/Transforms/SimdVectorizeLoops.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/TensorExt/Transforms/ImplementRotateAndReduce.h"
#include "lib/Utils/AffineLoopUtils.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/Sequence.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/SmallBitVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"       // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"         // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"           // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/LoopInvariantCodeMotionUtils.h"  // from @llvm-project

#define DEBUG_TYPE "simd-vectorize-loops"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_SIMDVECTORIZELOOPS
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

namespace {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::VarKind;

struct ConstantLoopBounds {
  int64_t lb;
  int64_t ub;
  int64_t step;

  int64_t tripCount() const {
    return ub <= lb ? 0 : (ub - lb + step - 1) / step;
  }
};

FailureOr<ConstantLoopBounds> getConstantLoopBounds(LoopLikeOpInterface loop) {
  std::optional<OpFoldResult> lb = loop.getSingleLowerBound();
  std::optional<OpFoldResult> ub = loop.getSingleUpperBound();
  std::optional<OpFoldResult> step = loop.getSingleStep();
  if (!lb || !ub || !step || !loop.getSingleInductionVar()) return failure();
  std::optional<int64_t> lbCst = getConstantIntValue(*lb);
  std::optional<int64_t> ubCst = getConstantIntValue(*ub);
  std::optional<int64_t> stepCst = getConstantIntValue(*step);
  if (!lbCst || !ubCst || !stepCst || *stepCst <= 0) return failure();
  return ConstantLoopBounds{*lbCst, *ubCst, *stepCst};
}

bool isSupportedLoop(Operation* op) {
  return isa<affine::AffineForOp, scf::ForOp>(op);
}

bool isReduction(Operation* op) {
  return isa<arith::AddIOp, arith::AddFOp, arith::MulIOp, arith::MulFOp>(op);
}

bool isMulReduction(OperationName name) {
  return name.getStringRef() == arith::MulIOp::getOperationName() ||
         name.getStringRef() == arith::MulFOp::getOperationName();
}

// Whether `value` is the identity of the reduction `name`.
bool isReductionIdentity(Value value, OperationName name) {
  if (isMulReduction(name))
    return matchPattern(value, m_One()) || matchPattern(value, m_OneFloat());
  return matchPattern(value, m_Zero()) || matchPattern(value, m_AnyZeroFloat());
}

bool usesAnyDim(AffineExpr expr, const DenseSet<unsigned>& dims) {
  bool found = false;
  expr.walk([&](AffineExpr e) {
    if (auto dim = dyn_cast<AffineDimExpr>(e))
      found |= dims.contains(dim.getPosition());
  });
  return found;
}

// Drop `mod m` from the top-level sum of `expr` when `m` is a multiple of
// `numSlots`. The result is congruent to `expr` modulo `numSlots`, which is all
// that matters for a cyclic rotation of a tensor with `numSlots` elements.
AffineExpr stripSlotModulus(AffineExpr expr, int64_t numSlots) {
  auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin) return expr;
  if (bin.getKind() == AffineExprKind::Mod) {
    auto modulus = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (modulus && modulus.getValue() % numSlots == 0)
      return stripSlotModulus(bin.getLHS(), numSlots);
    return expr;
  }
  if (bin.getKind() == AffineExprKind::Add)
    return stripSlotModulus(bin.getLHS(), numSlots) +
           stripSlotModulus(bin.getRHS(), numSlots);
  return expr;
}

// The set of slots written (or reduced over) by a loop nest, given by an
// affine map from the vectorized loops' induction variables to slot indices.
struct SlotMapInfo {
  // Whether the map is a bijection onto all slots.
  bool full = false;
  // The image of the map, only populated if !full.
  SmallVector<int64_t> image;
};

// Check that `slotExpr`, an affine function of the induction variables of
// `loops` (dimension i is the IV of loops[i]), injectively maps the iteration
// domain into [0, numSlots).
FailureOr<SlotMapInfo> analyzeSlotMap(ArrayRef<LoopLikeOpInterface> loops,
                                      AffineExpr slotExpr, int64_t numSlots) {
  FailureOr<affine::FlatAffineValueConstraints> domain =
      getIterationDomain(loops);
  if (failed(domain)) return failure();

  int64_t domainSize = 1;
  for (LoopLikeOpInterface loop : loops) {
    FailureOr<ConstantLoopBounds> bounds = getConstantLoopBounds(loop);
    if (failed(bounds)) return failure();
    domainSize *= bounds->tripCount();
  }

  unsigned numLoops = loops.size();
  affine::FlatAffineValueConstraints cst = std::move(*domain);
  unsigned slotPos = cst.appendDimVar();
  AffineMap slotMap = AffineMap::get(cst.getNumDimVars(), /*symbolCount=*/0,
                                     slotExpr, slotExpr.getContext());
  if (failed(cst.addBound(BoundType::EQ, slotPos, slotMap))) return failure();

  // loop IVs -> slot
  IntegerRelation rel(cst);
  rel.convertVarKind(VarKind::Range, 0, numLoops, VarKind::Domain);

  // The image must lie within [0, numSlots).
  unsigned slotCol = rel.getVarKindOffset(VarKind::Range);
  for (bool below : {true, false}) {
    IntegerRelation outOfRange = rel;
    SmallVector<int64_t> ineq(outOfRange.getNumCols(), 0);
    // below: -slot - 1 >= 0, above: slot - numSlots >= 0
    ineq[slotCol] = below ? -1 : 1;
    ineq.back() = below ? -1 : -numSlots;
    outOfRange.addInequality(ineq);
    if (!outOfRange.isIntegerEmpty()) {
      LDBG() << "slot map " << slotExpr << " leaves [0, " << numSlots << ")";
      return failure();
    }
  }

  // Injectivity: {(iv, iv') : slot(iv) = slot(iv')} must be the identity.
  IntegerRelation inverse = rel;
  inverse.inverse();
  IntegerRelation kernel = rel;
  kernel.compose(inverse);
  unsigned domainCol = kernel.getVarKindOffset(VarKind::Domain);
  unsigned rangeCol = kernel.getVarKindOffset(VarKind::Range);
  for (unsigned i = 0; i < numLoops; ++i) {
    for (int64_t sign : {1, -1}) {
      IntegerRelation differ = kernel;
      SmallVector<int64_t> ineq(differ.getNumCols(), 0);
      // sign * (iv_i - iv'_i) - 1 >= 0
      ineq[domainCol + i] = sign;
      ineq[rangeCol + i] = -sign;
      ineq.back() = -1;
      differ.addInequality(ineq);
      if (!differ.isIntegerEmpty()) {
        LDBG() << "slot map " << slotExpr << " is not injective";
        return failure();
      }
    }
  }

  SlotMapInfo info;
  info.full = domainSize == numSlots;
  if (!info.full) {
    PointCollector collector;
    getRangePoints(rel, collector);
    for (const auto& point : collector.points) info.image.push_back(point[0]);
  }
  return info;
}

// `expr`, a function of `numDims` dimensions, with dimension d replaced by v
// for each (d, v) in `values`
AffineExpr substituteDims(AffineExpr expr, unsigned numDims,
                          ArrayRef<std::pair<unsigned, int64_t>> values) {
  MLIRContext* ctx = expr.getContext();
  SmallVector<AffineExpr> replacements;
  for (unsigned i = 0; i < numDims; ++i)
    replacements.push_back(getAffineDimExpr(i, ctx));
  for (auto [dim, value] : values)
    replacements[dim] = getAffineConstantExpr(value, ctx);
  return simplifyAffineExpr(expr.replaceDims(replacements), numDims, 0);
}

// The most loop iterations enumerated, or plaintext table elements created, at
// compile time
constexpr int64_t kMaxPoints = 1 << 16;

// Call `fn` with each point of the box of loop iterations given by `bounds`
// (one per dimension in `dims`), as (dimension, value) pairs, in row-major
// order. Fails if the box has more than kMaxPoints points.
LogicalResult forEachPoint(
    ArrayRef<unsigned> dims, ArrayRef<ConstantLoopBounds> bounds,
    llvm::function_ref<void(ArrayRef<std::pair<unsigned, int64_t>>)> fn) {
  int64_t numPoints = 1;
  for (const ConstantLoopBounds& b : bounds) {
    numPoints *= b.tripCount();
    if (numPoints > kMaxPoints) return failure();
  }
  SmallVector<std::pair<unsigned, int64_t>> point(dims.size());
  for (int64_t n = 0; n < numPoints; ++n) {
    int64_t rest = n;
    for (int k = dims.size() - 1; k >= 0; --k) {
      point[k] = {dims[k], bounds[k].lb +
                               (rest % bounds[k].tripCount()) * bounds[k].step};
      rest /= bounds[k].tripCount();
    }
    fn(point);
  }
  return success();
}

// The row-major linear index of `indices` into a tensor of shape `shape`
int64_t linearize(ArrayRef<int64_t> indices, ArrayRef<int64_t> shape) {
  int64_t linear = 0;
  for (auto [index, size] : llvm::zip(indices, shape))
    linear = linear * size + index;
  return linear;
}

struct RecordingListener : public OpBuilder::Listener {
  SmallVector<Operation*> ops;
  void notifyOperationInserted(Operation* op,
                               OpBuilder::InsertPoint previous) override {
    ops.push_back(op);
  }
};

// A set of loop dimensions (indices into a nest's loops), as a bit mask.
using DimSet = uint64_t;

DimSet dimBit(unsigned dim) { return DimSet(1) << dim; }
bool hasDim(DimSet dims, unsigned dim) { return dims & dimBit(dim); }

// Whether `value` is known not to be secret, so that rearranging its elements
// does not require rotations.
bool isKnownPlaintext(Value value, const DataFlowSolver* solver) {
  if (matchPattern(value, m_Constant())) return true;
  const auto* lattice = solver->lookupState<SecretnessLattice>(value);
  return lattice && lattice->getValue().isInitialized() &&
         !lattice->getValue().getSecretness();
}

// Vectorizes a single loop nest rooted at `root`, using the index of
// `slotSource` as the slot map, or the insertion into the output of a map nest
// if it is null. See the description of the pass for the overall approach.
//
// Every scalar computed in the nest becomes a vector holding the scalar of each
// iteration of the vectorized loops in its slot. A value computed inside some
// of the vectorized loops (its active loops) is held in the slots given by the
// slot map with the IVs of the other vectorized loops at their lower bounds.
class NestVectorizer {
 public:
  NestVectorizer(LoopLikeOpInterface root, const DataFlowSolver* solver,
                 tensor::ExtractOp slotSource)
      : root(root),
        solver(solver),
        slotSource(slotSource),
        builder(root.getOperation()) {
    builder.setListener(&listener);
  }

  LogicalResult run();

 private:
  LogicalResult collectLoops();
  LogicalResult analyzeMapNest();
  LogicalResult configure();

  LogicalResult processBlock(Block& block);
  LogicalResult processOp(Operation* op);
  LogicalResult processMapLoop(LoopLikeOpInterface loop);
  LogicalResult processReductionLoop(LoopLikeOpInterface loop);
  LogicalResult processKeptLoop(LoopLikeOpInterface loop);
  LogicalResult processExtract(tensor::ExtractOp op);
  LogicalResult processElementwise(Operation* op);

  LogicalResult finalizeMapNest();
  LogicalResult finalizeReductionNest();
  void foldKeptLoops();
  void rollback();

  bool isDefinedInNest(Value value) const;
  bool isKnownPlaintext(Value value) const {
    return ::mlir::heir::tensor_ext::isKnownPlaintext(value, solver);
  }
  unsigned getDim(LoopLikeOpInterface loop) const {
    return llvm::find(allLoops, loop) - allLoops.begin();
  }
  bool isVectorizedLoop(LoopLikeOpInterface loop) const {
    unsigned dim = getDim(loop);
    return dim < allLoops.size() && hasDim(vectorizedDims, dim);
  }
  SmallVector<unsigned> enclosingDims(Operation* op) const;
  DimSet activeDimsAt(Operation* op) const;
  AffineExpr slotAt(DimSet active) const;
  AffineExpr evaluate(AffineExpr expr,
                      ArrayRef<std::pair<unsigned, int64_t>> values) const;
  AffineExpr renumber(AffineExpr expr,
                      ArrayRef<LoopLikeOpInterface> loops) const;
  FailureOr<OperationName> getReductionKind(BlockArgument acc) const;

  FailureOr<Value> getScalar(Value value);
  FailureOr<Value> getVector(Value value, DimSet active);
  void setVector(Value oldValue, Value vector, DimSet active);
  Value splat(Value scalar);
  Value flatten(Value tensor);
  Value materializeIndex(AffineExpr expr,
                         const DenseMap<unsigned, Value>& dimValues = {});
  Value createRotation(Value tensor, AffineExpr shift);
  Value createMask(Type elementType, int64_t size, ArrayRef<int64_t> slots,
                   bool inverted);
  Value createArith(bool multiply, Value lhs, Value rhs);
  Value createReduction(OperationName name, Value lhs, Value rhs);
  FailureOr<Value> createGather(tensor::ExtractOp op, Value tensor,
                                AffineExpr flatIndex, DimSet active);
  FailureOr<Value> relayout(Value vector, AffineExpr from, int64_t toSize,
                            AffineExpr to, DimSet domain);
  FailureOr<Value> reduceOverLoop(Value vector, unsigned dim,
                                  OperationName name);
  FailureOr<std::pair<Value, AffineExpr>> resolveExtract(
      tensor::ExtractOp op) const;
  FailureOr<AffineExpr> getInvariantOffset(AffineExpr index, DimSet active);
  FailureOr<ConstantLoopBounds> getDimBounds(unsigned dim) const;

  LoopLikeOpInterface root;
  // The op before `root` when vectorization started (null if `root` was the
  // first op in its block). New code is inserted between it and `root`.
  Operation* rootPredecessor = nullptr;
  const DataFlowSolver* solver;
  // The extraction whose index is the slot map, or null to use the insertion
  // into the output tensor of a map nest.
  tensor::ExtractOp slotSource;
  OpBuilder builder;
  RecordingListener listener;

  // All loops in the nest in pre-order, and their induction variables. An
  // affine expression's dimension i refers to allIVs[i].
  SmallVector<LoopLikeOpInterface> allLoops;
  SmallVector<Value> allIVs;

  // For map nests, the loops carrying the output tensor (outermost first), the
  // insertion into it, and its flattened index.
  SmallVector<LoopLikeOpInterface> mapLoops;
  tensor::InsertOp mapInsert;
  AffineExpr outputIndex;
  int64_t numOutputSlots = 0;

  // The vectorized loops in pre-order
  SmallVector<LoopLikeOpInterface> vectorizedLoops;
  DimSet vectorizedDims = 0;

  int64_t numSlots = 0;
  AffineExpr slotExpr;

  // Old scalar value -> new vector, and the active loops of the scalar
  DenseMap<Value, Value> vectors;
  DenseMap<Value, DimSet> vectorDims;
  // (old scalar value, active loops) -> the vector relaid out for these loops
  DenseMap<std::pair<Value, DimSet>, Value> relaidOut;
  // New vectors that hold the same value in every slot
  DenseSet<Value> uniformVectors;
  // New vectors that are known to be plaintexts
  DenseSet<Value> plaintextVectors;
  // Iter args of vectorized loops -> the op reducing into them. Inside the
  // loop, they stand for the identity of the reduction.
  DenseMap<Value, OperationName> reductionArgs;
  // rotate_and_reduce ops to implement once the nest is vectorized
  SmallVector<RotateAndReduceOp> pendingReductions;
  // Old IV of a kept loop -> new IV
  DenseMap<Value, Value> ivMapping;
  DenseMap<Value, Value> flatTensors;
  SmallVector<LoopLikeOpInterface> keptLoops;
};

bool NestVectorizer::isDefinedInNest(Value value) const {
  Operation* owner = isa<BlockArgument>(value)
                         ? cast<BlockArgument>(value).getOwner()->getParentOp()
                         : value.getDefiningOp();
  return root->isAncestor(owner);
}

LogicalResult NestVectorizer::collectLoops() {
  WalkResult result = root->walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
      if (!isSupportedLoop(op) || !loop.getSingleInductionVar())
        return WalkResult::interrupt();
      allLoops.push_back(loop);
      allIVs.push_back(*loop.getSingleInductionVar());
      return WalkResult::advance();
    }
    if (op->getNumRegions() != 0) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted() ||
                 allLoops.size() > 8 * sizeof(DimSet));
}

LogicalResult NestVectorizer::analyzeMapNest() {
  LoopLikeOpInterface current = root;
  Type tensorType;
  SmallVector<LoopLikeOpInterface> loops;
  while (true) {
    if (current->getNumResults() != 1 ||
        current.getRegionIterArgs().size() != 1)
      return failure();
    BlockArgument iterArg = current.getRegionIterArgs()[0];
    if (!isa<RankedTensorType>(iterArg.getType())) return failure();
    if (tensorType && iterArg.getType() != tensorType) return failure();
    tensorType = iterArg.getType();
    if (failed(getConstantLoopBounds(current)) || !iterArg.hasOneUse())
      return failure();

    loops.push_back(current);
    Value yielded = current.getYieldedValues()[0];
    Operation* user = *iterArg.getUsers().begin();
    if (auto nested = dyn_cast<LoopLikeOpInterface>(user)) {
      if (nested->getParentOp() != current.getOperation() ||
          yielded != nested->getResult(0) || nested.getInits().size() != 1 ||
          nested.getInits()[0] != iterArg)
        return failure();
      current = nested;
      continue;
    }
    auto insert = dyn_cast<tensor::InsertOp>(user);
    if (!insert || insert.getDest() != iterArg ||
        yielded != insert.getResult() || !insert.getResult().hasOneUse())
      return failure();

    auto outputType = cast<RankedTensorType>(tensorType);
    if (!outputType.hasStaticShape()) return failure();
    // Flattened (row-major) insertion index
    AffineExpr flatIndex = getAffineConstantExpr(0, root->getContext());
    for (auto [index, dimSize] :
         llvm::zip(insert.getIndices(), outputType.getShape())) {
      FailureOr<AffineExpr> expr = getAffineExprOfIVs(index, allIVs);
      if (failed(expr)) return failure();
      flatIndex = flatIndex * dimSize + *expr;
    }
    mapLoops = std::move(loops);
    mapInsert = insert;
    outputIndex = flatIndex;
    numOutputSlots = outputType.getNumElements();
    return success();
  }
}

// Whether `acc`, the iter arg of a loop, only accumulates a reduction: it is
// combined once with a value that does not depend on it, or passed to a nested
// loop that does so, and the result is yielded. Returns the reduction op.
FailureOr<OperationName> NestVectorizer::getReductionKind(
    BlockArgument acc) const {
  auto loop = dyn_cast<LoopLikeOpInterface>(acc.getOwner()->getParentOp());
  if (!loop || loop.getRegionIterArgs().size() != 1 || !acc.hasOneUse())
    return failure();
  Operation* user = *acc.getUsers().begin();
  std::optional<OperationName> name;
  Value result;
  if (isReduction(user)) {
    name = user->getName();
    result = user->getResult(0);
  } else if (auto nested = dyn_cast<LoopLikeOpInterface>(user)) {
    if (nested.getInits().size() != 1 || nested.getInits()[0] != acc)
      return failure();
    FailureOr<OperationName> nestedName =
        getReductionKind(nested.getRegionIterArgs()[0]);
    if (failed(nestedName)) return failure();
    name = *nestedName;
    result = nested->getResult(0);
  } else {
    return failure();
  }
  if (!result.hasOneUse() || loop.getYieldedValues()[0] != result)
    return failure();
  return *name;
}

LogicalResult NestVectorizer::configure() {
  if (slotSource) {
    FailureOr<std::pair<Value, AffineExpr>> resolved =
        resolveExtract(slotSource);
    if (failed(resolved)) return failure();
    auto type = cast<RankedTensorType>(resolved->first.getType());
    if (!type.hasStaticShape()) return failure();
    numSlots = type.getNumElements();
    slotExpr = resolved->second;
  } else {
    if (!mapInsert) return failure();
    numSlots = numOutputSlots;
    slotExpr = outputIndex;
  }

  for (auto [dim, loop] : llvm::enumerate(allLoops)) {
    bool isMapLoop = llvm::is_contained(mapLoops, loop);
    if (!isMapLoop && !slotExpr.isFunctionOfDim(dim)) continue;
    if (failed(getConstantLoopBounds(loop))) return failure();
    if (!isMapLoop) {
      // The loop must reduce into a scalar.
      if (loop.getRegionIterArgs().size() != 1) return failure();
      Type type = loop.getRegionIterArgs()[0].getType();
      if (isa<ShapedType>(type) || type.isIndex()) return failure();
    }
    vectorizedLoops.push_back(loop);
    vectorizedDims |= dimBit(dim);
  }
  if (!mapInsert && !isVectorizedLoop(root)) return failure();
  if (failed(analyzeSlotMap(vectorizedLoops,
                            renumber(slotExpr, vectorizedLoops), numSlots)))
    return failure();
  LDBG() << "vectorizing " << vectorizedLoops.size() << " of "
         << allLoops.size() << " loops with slot map " << slotExpr << " into "
         << numSlots << " slots: " << root;
  return success();
}

// The dimensions of the loops of the nest enclosing `op`, outermost first
SmallVector<unsigned> NestVectorizer::enclosingDims(Operation* op) const {
  SmallVector<unsigned> dims;
  for (Operation* parent = op->getParentOp();
       parent && parent != root->getParentOp();
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<LoopLikeOpInterface>(parent);
    if (loop && getDim(loop) < allLoops.size()) dims.push_back(getDim(loop));
  }
  std::reverse(dims.begin(), dims.end());
  return dims;
}

// The vectorized loops enclosing `op`
DimSet NestVectorizer::activeDimsAt(Operation* op) const {
  DimSet active = 0;
  for (unsigned dim : enclosingDims(op))
    if (hasDim(vectorizedDims, dim)) active |= dimBit(dim);
  return active;
}

AffineExpr NestVectorizer::evaluate(
    AffineExpr expr, ArrayRef<std::pair<unsigned, int64_t>> values) const {
  return substituteDims(expr, allIVs.size(), values);
}

// The slots of values whose active loops are `active`: the slot map with the
// IVs of the other vectorized loops at their lower bounds.
AffineExpr NestVectorizer::slotAt(DimSet active) const {
  SmallVector<std::pair<unsigned, int64_t>> values;
  for (unsigned dim = 0; dim < allLoops.size(); ++dim)
    if (hasDim(vectorizedDims, dim) && !hasDim(active, dim))
      values.push_back({dim, getConstantLoopBounds(allLoops[dim])->lb});
  return evaluate(slotExpr, values);
}

// Renumber the dimensions of `expr` so that dimension i is the IV of loops[i].
// `expr` may only depend on these loops' IVs.
AffineExpr NestVectorizer::renumber(AffineExpr expr,
                                    ArrayRef<LoopLikeOpInterface> loops) const {
  MLIRContext* ctx = expr.getContext();
  SmallVector<AffineExpr> replacements;
  for (LoopLikeOpInterface loop : allLoops) {
    const auto* it = llvm::find(loops, loop);
    replacements.push_back(it == loops.end()
                               ? getAffineConstantExpr(0, ctx)
                               : getAffineDimExpr(it - loops.begin(), ctx));
  }
  return expr.replaceDims(replacements);
}

FailureOr<Value> NestVectorizer::getScalar(Value value) {
  if (auto it = ivMapping.find(value); it != ivMapping.end()) return it->second;
  if (!isDefinedInNest(value)) return value;
  if (const auto* it = llvm::find(allIVs, value);
      it != allIVs.end() && hasDim(vectorizedDims, it - allIVs.begin()))
    return failure();
  if (vectors.contains(value)) return failure();
  Operation* def = value.getDefiningOp();
  if (!def || def->getNumRegions() != 0 || !isMemoryEffectFree(def))
    return failure();
  IRMapping mapping;
  for (Value operand : def->getOperands()) {
    FailureOr<Value> scalar = getScalar(operand);
    if (failed(scalar)) return failure();
    mapping.map(operand, *scalar);
  }
  Operation* clone = builder.clone(*def, mapping);
  return clone->getResult(cast<OpResult>(value).getResultNumber());
}

Value NestVectorizer::splat(Value scalar) {
  auto type = RankedTensorType::get({numSlots}, scalar.getType());
  Value splatted =
      tensor::SplatOp::create(builder, root.getLoc(), scalar, type);
  uniformVectors.insert(splatted);
  return splatted;
}

// The vector for the old scalar `value` in the slots of values with active
// loops `active`.
FailureOr<Value> NestVectorizer::getVector(Value value, DimSet active) {
  auto it = vectors.find(value);
  if (it == vectors.end()) {
    FailureOr<Value> scalar = getScalar(value);
    if (failed(scalar)) return failure();
    Value splatted = splat(*scalar);
    if (isKnownPlaintext(value)) plaintextVectors.insert(splatted);
    return splatted;
  }

  Value vector = it->second;
  DimSet dims = vectorDims.lookup(value);
  if (uniformVectors.contains(vector) || dims == active) return vector;
  // A value can only be used inside its active loops.
  if (dims & ~active) return failure();
  if (Value cached = relaidOut.lookup({value, active})) return cached;
  FailureOr<Value> moved =
      relayout(vector, slotAt(dims), numSlots, slotAt(active), active);
  if (failed(moved)) return failure();
  if (plaintextVectors.contains(vector)) plaintextVectors.insert(*moved);
  relaidOut[{value, active}] = *moved;
  return *moved;
}

void NestVectorizer::setVector(Value oldValue, Value vector, DimSet active) {
  vectors[oldValue] = vector;
  vectorDims[oldValue] = active;
}

// `tensor` as a one-dimensional tensor
Value NestVectorizer::flatten(Value tensor) {
  if (Value flat = flatTensors.lookup(tensor)) return flat;
  auto type = cast<RankedTensorType>(tensor.getType());
  Value flat = tensor;
  if (type.getRank() != 1) {
    OpBuilder::InsertionGuard guard(builder);
    if (auto arg = dyn_cast<BlockArgument>(tensor))
      builder.setInsertionPointToStart(arg.getOwner());
    else
      builder.setInsertionPointAfter(tensor.getDefiningOp());
    ReassociationIndices allDims =
        llvm::to_vector(llvm::seq<int64_t>(0, type.getRank()));
    flat = tensor::CollapseShapeOp::create(
        builder, root.getLoc(),
        RankedTensorType::get({type.getNumElements()}, type.getElementType()),
        tensor, ArrayRef<ReassociationIndices>{allDims});
  }
  flatTensors[tensor] = flat;
  return flat;
}

// Materialize `expr` at the current insertion point, where the dimensions in
// `dimValues` take the given values, and the IVs of kept loops are mapped to
// the IVs of the new loops.
Value NestVectorizer::materializeIndex(
    AffineExpr expr, const DenseMap<unsigned, Value>& dimValues) {
  Location loc = root.getLoc();
  if (auto cst = dyn_cast<AffineConstantExpr>(expr))
    return arith::ConstantIndexOp::create(builder, loc, cst.getValue());
  SmallVector<Value> operands;
  for (auto [dim, iv] : llvm::enumerate(allIVs)) {
    if (Value value = dimValues.lookup(dim)) {
      operands.push_back(value);
      continue;
    }
    // Other IVs do not appear in `expr`.
    operands.push_back(ivMapping.lookup(iv));
  }
  return affine::expandAffineExpr(builder, loc, expr, operands, {});
}

Value NestVectorizer::createRotation(Value tensor, AffineExpr shift) {
  int64_t size = cast<RankedTensorType>(tensor.getType()).getNumElements();
  shift = simplifyAffineExpr(shift, allIVs.size(), 0);
  if (auto cst = dyn_cast<AffineConstantExpr>(shift)) {
    int64_t normalized = ((cst.getValue() % size) + size) % size;
    if (normalized == 0) return tensor;
    shift = getAffineConstantExpr(normalized, shift.getContext());
  }
  return RotateOp::create(builder, root.getLoc(), tensor,
                          materializeIndex(shift));
}

Value NestVectorizer::createMask(Type elementType, int64_t size,
                                 ArrayRef<int64_t> slots, bool inverted) {
  auto type = RankedTensorType::get({size}, elementType);
  Attribute zero = builder.getZeroAttr(elementType);
  Attribute one = builder.getOneAttr(elementType);
  SmallVector<Attribute> values(size, inverted ? one : zero);
  for (int64_t slot : slots) values[slot] = inverted ? zero : one;
  Value mask = arith::ConstantOp::create(
      builder, root.getLoc(),
      cast<TypedAttr>(DenseElementsAttr::get(type, values)));
  plaintextVectors.insert(mask);
  return mask;
}

Value NestVectorizer::createArith(bool multiply, Value lhs, Value rhs) {
  Location loc = root.getLoc();
  if (isa<FloatType>(getElementTypeOrSelf(lhs.getType()))) {
    if (multiply) return arith::MulFOp::create(builder, loc, lhs, rhs);
    return arith::AddFOp::create(builder, loc, lhs, rhs);
  }
  if (multiply) return arith::MulIOp::create(builder, loc, lhs, rhs);
  return arith::AddIOp::create(builder, loc, lhs, rhs);
}

Value NestVectorizer::createReduction(OperationName name, Value lhs,
                                      Value rhs) {
  OperationState state(root.getLoc(), name);
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType());
  return builder.create(state)->getResult(0);
}

// The vector for the extraction `op` from `tensor`, a plaintext, at
// `flatIndex`: its slot for each iteration of the vectorized loops in `active`
// holds the extracted element, and other slots hold zero. The vectors for all
// iterations of the kept loops enclosing `op` form a table indexed by these
// loops, which is built once, before the enclosing secret.generic if `tensor`
// is defined outside it and before the vectorized code otherwise: as a constant
// if `tensor` is a constant, and by a loop nest otherwise. The result is the
// row of the table for the current iteration of the kept loops.
FailureOr<Value> NestVectorizer::createGather(tensor::ExtractOp op,
                                              Value tensor,
                                              AffineExpr flatIndex,
                                              DimSet active) {
  auto type = cast<RankedTensorType>(tensor.getType());
  if (!type.hasStaticShape()) return failure();
  MLIRContext* ctx = op.getContext();
  Location loc = root.getLoc();
  Type elementType = op.getResult().getType();

  SmallVector<unsigned> keptDims;
  for (unsigned dim : enclosingDims(op))
    if (!hasDim(vectorizedDims, dim)) keptDims.push_back(dim);
  SmallVector<unsigned> tableDims = keptDims;
  for (unsigned dim = 0; dim < allLoops.size(); ++dim)
    if (hasDim(active, dim)) tableDims.push_back(dim);
  SmallVector<ConstantLoopBounds> bounds;
  for (unsigned dim : tableDims) {
    FailureOr<ConstantLoopBounds> b = getDimBounds(dim);
    if (failed(b)) return failure();
    bounds.push_back(*b);
  }
  SmallVector<int64_t> shape;
  SmallVector<AffineExpr> rows;
  for (auto [dim, b] : llvm::zip(keptDims, bounds)) {
    shape.push_back(b.tripCount());
    rows.push_back((getAffineDimExpr(dim, ctx) - b.lb).floorDiv(b.step));
  }
  shape.push_back(numSlots);
  int64_t tableSize = std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                      std::multiplies<int64_t>());
  if (tableSize > kMaxPoints) return failure();
  auto tableType = RankedTensorType::get(shape, elementType);
  AffineExpr slot = slotAt(active);

  // A non-secret operand of the enclosing generic is available before it.
  Value source = tensor;
  auto generic = root->getParentOfType<secret::GenericOp>();
  if (auto arg = dyn_cast<BlockArgument>(source);
      generic && arg && arg.getOwner() == generic.getBody())
    source = generic.getOpOperandForBlockArgument(arg)->get();
  bool beforeGeneric =
      generic && !generic->isAncestor(source.getParentBlock()->getParentOp());

  Value table;
  {
    OpBuilder::InsertionGuard guard(builder);
    if (beforeGeneric)
      builder.setInsertionPoint(generic);
    else if (rootPredecessor)
      builder.setInsertionPointAfter(rootPredecessor);
    else
      builder.setInsertionPointToStart(root->getBlock());
    DenseElementsAttr values;
    if (matchPattern(source, m_Constant(&values))) {
      SmallVector<Attribute> entries(tableSize,
                                     builder.getZeroAttr(elementType));
      auto sourceValues = values.getValues<Attribute>();
      bool inBounds = true;
      LogicalResult enumerated = forEachPoint(
          tableDims, bounds, [&](ArrayRef<std::pair<unsigned, int64_t>> point) {
            auto flat =
                dyn_cast<AffineConstantExpr>(evaluate(flatIndex, point));
            auto slotIndex =
                dyn_cast<AffineConstantExpr>(evaluate(slot, point));
            SmallVector<int64_t> index;
            for (AffineExpr row : rows)
              index.push_back(
                  cast<AffineConstantExpr>(evaluate(row, point)).getValue());
            if (!flat || !slotIndex || flat.getValue() < 0 ||
                flat.getValue() >= type.getNumElements()) {
              inBounds = false;
              return;
            }
            index.push_back(slotIndex.getValue());
            entries[linearize(index, shape)] = sourceValues[flat.getValue()];
          });
      if (failed(enumerated) || !inBounds) return failure();
      table = arith::ConstantOp::create(
          builder, loc,
          cast<TypedAttr>(DenseElementsAttr::get(tableType, entries)));
    } else {
      Value flat = flatten(source);
      Value zeros = arith::ConstantOp::create(
          builder, loc,
          cast<TypedAttr>(DenseElementsAttr::get(
              tableType, builder.getZeroAttr(elementType))));
      DenseMap<unsigned, Value> dimValues;
      std::function<Value(unsigned, Value)> buildLevel =
          [&](unsigned level, Value acc) -> Value {
        if (level == tableDims.size()) {
          Value element = tensor::ExtractOp::create(
              builder, loc, flat, materializeIndex(flatIndex, dimValues));
          SmallVector<Value> indices;
          for (AffineExpr row : rows)
            indices.push_back(materializeIndex(row, dimValues));
          indices.push_back(materializeIndex(slot, dimValues));
          return tensor::InsertOp::create(builder, loc, element, acc, indices);
        }
        auto forOp = affine::AffineForOp::create(builder, loc, bounds[level].lb,
                                                 bounds[level].ub,
                                                 bounds[level].step, acc);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(forOp.getBody());
        dimValues[tableDims[level]] = forOp.getInductionVar();
        Value result = buildLevel(level + 1, forOp.getRegionIterArgs()[0]);
        affine::AffineYieldOp::create(builder, loc, result);
        return forOp.getResult(0);
      };
      table = buildLevel(0, zeros);
    }
  }
  plaintextVectors.insert(table);
  if (keptDims.empty()) return table;

  // The row of the current iteration of the kept loops
  SmallVector<OpFoldResult> offsets;
  for (auto [dim, b] : llvm::zip(keptDims, bounds)) {
    AffineMap row = AffineMap::get(
        1, 0, (getAffineDimExpr(0, ctx) - b.lb).floorDiv(b.step));
    offsets.push_back(affine::AffineApplyOp::create(
                          builder, loc, row, ivMapping.lookup(allIVs[dim]))
                          .getResult());
  }
  offsets.push_back(builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes(keptDims.size(), builder.getIndexAttr(1));
  sizes.push_back(builder.getIndexAttr(numSlots));
  SmallVector<OpFoldResult> strides(shape.size(), builder.getIndexAttr(1));
  Value row = tensor::ExtractSliceOp::create(
      builder, loc, RankedTensorType::get({numSlots}, elementType), table,
      offsets, sizes, strides);
  plaintextVectors.insert(row);
  return row;
}

// Move the value of each iteration of the loops in `domain` from slot `from` of
// `vector` to slot `to` of a vector with `toSize` slots, where `from` and `to`
// are affine functions of these loops' IVs. Iterations with the same offset
// between the two slots share one rotation, followed by a mask selecting their
// slots unless a single rotation moves all slots. Slots not written by any
// iteration hold zero.
FailureOr<Value> NestVectorizer::relayout(Value vector, AffineExpr from,
                                          int64_t toSize, AffineExpr to,
                                          DimSet domain) {
  auto type = cast<RankedTensorType>(vector.getType());
  int64_t fromSize = type.getNumElements();
  int64_t width = std::max(fromSize, toSize);

  SmallVector<unsigned> dims;
  SmallVector<ConstantLoopBounds> bounds;
  for (unsigned dim = 0; dim < allLoops.size(); ++dim) {
    if (!hasDim(domain, dim)) continue;
    FailureOr<ConstantLoopBounds> b = getDimBounds(dim);
    if (failed(b)) return failure();
    dims.push_back(dim);
    bounds.push_back(*b);
  }

  // offset -> destination slots
  std::map<int64_t, SmallVector<int64_t>> classes;
  bool inRange = true;
  LogicalResult enumerated = forEachPoint(
      dims, bounds, [&](ArrayRef<std::pair<unsigned, int64_t>> point) {
        auto fromSlot = dyn_cast<AffineConstantExpr>(evaluate(from, point));
        auto toSlot = dyn_cast<AffineConstantExpr>(evaluate(to, point));
        if (!fromSlot || !toSlot || fromSlot.getValue() < 0 ||
            fromSlot.getValue() >= fromSize || toSlot.getValue() < 0 ||
            toSlot.getValue() >= toSize) {
          inRange = false;
          return;
        }
        int64_t offset =
            ((fromSlot.getValue() - toSlot.getValue()) % width + width) % width;
        classes[offset].push_back(toSlot.getValue());
      });
  if (failed(enumerated) || !inRange || classes.empty()) return failure();

  Location loc = root.getLoc();
  Type elementType = type.getElementType();
  Value source = vector;
  if (fromSize < width) {
    Value zeros = arith::ConstantOp::create(
        builder, loc,
        cast<TypedAttr>(
            DenseElementsAttr::get(RankedTensorType::get({width}, elementType),
                                   builder.getZeroAttr(elementType))));
    source = tensor::InsertSliceOp::create(
        builder, loc, vector, zeros,
        ArrayRef<OpFoldResult>{builder.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{builder.getIndexAttr(fromSize)},
        ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
  }

  Value result;
  for (auto& [offset, slots] : classes) {
    Value rotated = createRotation(
        source, getAffineConstantExpr(offset, root->getContext()));
    if (classes.size() != 1 || static_cast<int64_t>(slots.size()) != width)
      rotated = createArith(/*multiply=*/true, rotated,
                            createMask(elementType, width, slots, false));
    result =
        result ? createArith(/*multiply=*/false, result, rotated) : rotated;
  }
  if (toSize < width) {
    result = tensor::ExtractSliceOp::create(
        builder, loc, result, ArrayRef<OpFoldResult>{builder.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{builder.getIndexAttr(toSize)},
        ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
  }
  LDBG() << "relaid out with " << classes.size() << " rotations: " << from
         << " -> " << to;
  return result;
}

// Reduce `vector` over the slots of the iterations of the vectorized loop with
// dimension `dim`, leaving the result in the slots of its first iteration.
FailureOr<Value> NestVectorizer::reduceOverLoop(Value vector, unsigned dim,
                                                OperationName name) {
  unsigned numDims = allIVs.size();
  // The slot map must be stride * iv + rest, with rest independent of iv.
  AffineExpr rest = evaluate(slotExpr, {{dim, 0}});
  auto stride = dyn_cast<AffineConstantExpr>(
      simplifyAffineExpr(evaluate(slotExpr, {{dim, 1}}) - rest, numDims, 0));
  if (!stride ||
      simplifyAffineExpr(
          slotExpr - rest - stride * getAffineDimExpr(dim, root->getContext()),
          numDims, 0) != getAffineConstantExpr(0, root->getContext()))
    return failure();
  ConstantLoopBounds bounds = *getConstantLoopBounds(allLoops[dim]);
  int64_t period = stride.getValue() * bounds.step;
  int64_t steps = bounds.tripCount();
  if (steps == 1) return vector;
  if (period <= 0 || period > numSlots || steps > numSlots) return failure();

  auto reduction =
      RotateAndReduceOp::create(builder, root.getLoc(), vector, Value(), period,
                                steps, name.getStringRef());
  pendingReductions.push_back(reduction);
  return reduction.getResult();
}

// Resolve the tensor extracted from by `op` through slices taken inside the
// nest, and the flattened (row-major) index into that tensor as an affine
// function of the nest's IVs.
FailureOr<std::pair<Value, AffineExpr>> NestVectorizer::resolveExtract(
    tensor::ExtractOp op) const {
  Value tensor = op.getTensor();
  SmallVector<AffineExpr> indices;
  for (Value index : op.getIndices()) {
    FailureOr<AffineExpr> expr = getAffineExprOfIVs(index, allIVs);
    if (failed(expr)) return failure();
    indices.push_back(*expr);
  }
  while (isDefinedInNest(tensor)) {
    auto slice = tensor.getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice || !slice.hasUnitStride()) return failure();
    llvm::SmallBitVector droppedDims = slice.getDroppedDims();
    SmallVector<AffineExpr> sourceIndices;
    unsigned resultDim = 0;
    for (auto [i, offset] : llvm::enumerate(slice.getMixedOffsets())) {
      AffineExpr offsetExpr;
      if (std::optional<int64_t> cst = getConstantIntValue(offset)) {
        offsetExpr = getAffineConstantExpr(*cst, op.getContext());
      } else {
        FailureOr<AffineExpr> expr =
            getAffineExprOfIVs(cast<Value>(offset), allIVs);
        if (failed(expr)) return failure();
        offsetExpr = *expr;
      }
      if (droppedDims.test(i)) {
        sourceIndices.push_back(offsetExpr);
      } else {
        sourceIndices.push_back(offsetExpr + indices[resultDim++]);
      }
    }
    indices = std::move(sourceIndices);
    tensor = slice.getSource();
  }

  auto tensorType = cast<RankedTensorType>(tensor.getType());
  AffineExpr flatIndex = getAffineConstantExpr(0, op.getContext());
  for (auto [index, dimSize] : llvm::zip(indices, tensorType.getShape()))
    flatIndex = flatIndex * dimSize + index;
  return std::make_pair(tensor, flatIndex);
}

FailureOr<ConstantLoopBounds> NestVectorizer::getDimBounds(unsigned dim) const {
  return getConstantLoopBounds(allLoops[dim]);
}

// The offset of `index`, the flattened index of an extraction from a tensor
// with one element per slot, from the slots of values with active loops
// `active`, if it is the same for all iterations of these loops modulo the
// number of slots.
FailureOr<AffineExpr> NestVectorizer::getInvariantOffset(AffineExpr index,
                                                         DimSet active) {
  unsigned numIVs = allIVs.size();
  AffineExpr offset =
      simplifyAffineExpr(stripSlotModulus(index, numSlots) -
                             stripSlotModulus(slotAt(active), numSlots),
                         numIVs, 0);
  DenseSet<unsigned> activePositions;
  for (unsigned dim = 0; dim < numIVs; ++dim)
    if (hasDim(active, dim)) activePositions.insert(dim);
  if (!usesAnyDim(offset, activePositions)) return offset;

  // The offset may still be invariant modulo the number of slots, e.g., for
  // 4 * ((i + 1) mod 4) - 4 * i in a 16-slot tensor. Check that
  // {(iv, iv') : offset(iv) != offset(iv') mod numSlots}, where iv' agrees with
  // iv on the inactive loops, is empty.
  MLIRContext* ctx = index.getContext();
  SmallVector<unsigned> positions = llvm::to_vector(activePositions);
  llvm::sort(positions);
  unsigned numCopies = positions.size();
  unsigned lhsPos = numIVs + numCopies;
  unsigned rhsPos = lhsPos + 1;
  affine::FlatAffineValueConstraints cst(rhsPos + 1, /*numSymbols=*/0);

  auto addDomain = [&](unsigned pos, const ConstantLoopBounds& bounds) {
    cst.addBound(BoundType::LB, pos, bounds.lb);
    cst.addBound(BoundType::UB, pos, bounds.ub - 1);
    if (bounds.step == 1) return;
    SmallVector<int64_t> dividend(cst.getNumCols(), 0);
    dividend[pos] = 1;
    dividend.back() = -bounds.lb;
    unsigned quotientPos = cst.addLocalFloorDiv(dividend, bounds.step);
    SmallVector<int64_t> eq(cst.getNumCols(), 0);
    eq[pos] = 1;
    eq[quotientPos] = -bounds.step;
    eq.back() = -bounds.lb;
    cst.addEquality(eq);
  };

  SmallVector<ConstantLoopBounds> allBounds;
  for (unsigned i = 0; i < numIVs; ++i) {
    FailureOr<ConstantLoopBounds> bounds = getDimBounds(i);
    if (failed(bounds)) return failure();
    allBounds.push_back(*bounds);
  }
  for (unsigned i = 0; i < numIVs; ++i) addDomain(i, allBounds[i]);
  SmallVector<AffineExpr> copyReplacements;
  SmallVector<AffineExpr> lowerBoundReplacements;
  for (unsigned i = 0; i < numIVs; ++i) {
    const auto* it = llvm::find(positions, i);
    if (it == positions.end()) {
      copyReplacements.push_back(getAffineDimExpr(i, ctx));
      lowerBoundReplacements.push_back(getAffineDimExpr(i, ctx));
      continue;
    }
    unsigned copy = numIVs + (it - positions.begin());
    addDomain(copy, allBounds[i]);
    copyReplacements.push_back(getAffineDimExpr(copy, ctx));
    lowerBoundReplacements.push_back(
        getAffineConstantExpr(allBounds[i].lb, ctx));
  }

  AffineExpr lhs = offset % numSlots;
  AffineExpr rhs = offset.replaceDims(copyReplacements) % numSlots;
  if (failed(cst.addBound(BoundType::EQ, lhsPos,
                          AffineMap::get(rhsPos + 1, 0, lhs, ctx))) ||
      failed(cst.addBound(BoundType::EQ, rhsPos,
                          AffineMap::get(rhsPos + 1, 0, rhs, ctx))))
    return failure();
  for (int64_t sign : {1, -1}) {
    affine::FlatAffineValueConstraints differ = cst;
    SmallVector<int64_t> ineq(differ.getNumCols(), 0);
    // sign * (lhs - rhs) - 1 >= 0
    ineq[lhsPos] = sign;
    ineq[rhsPos] = -sign;
    ineq.back() = -1;
    differ.addInequality(ineq);
    if (!differ.isIntegerEmpty()) return failure();
  }

  // Evaluate the offset at the first active iteration.
  return simplifyAffineExpr(offset.replaceDims(lowerBoundReplacements), numIVs,
                            0);
}

LogicalResult NestVectorizer::processExtract(tensor::ExtractOp op) {
  FailureOr<std::pair<Value, AffineExpr>> resolved = resolveExtract(op);
  if (failed(resolved)) {
    // Not an affine access; fine as long as it is only used as a scalar.
    return success();
  }
  auto [tensor, flatIndex] = *resolved;
  DimSet active = activeDimsAt(op);
  DenseSet<unsigned> activePositions;
  for (unsigned dim = 0; dim < allIVs.size(); ++dim)
    if (hasDim(active, dim)) activePositions.insert(dim);
  if (!usesAnyDim(flatIndex, activePositions)) {
    // A scalar for every vectorized iteration, materialized when used.
    return success();
  }

  // Only a tensor with one element per slot can be aligned by a rotation.
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  if (!tensorType.hasStaticShape()) return failure();
  if (tensorType.getNumElements() == numSlots) {
    FailureOr<AffineExpr> offset = getInvariantOffset(flatIndex, active);
    if (succeeded(offset)) {
      setVector(op.getResult(), createRotation(flatten(tensor), *offset),
                active);
      return success();
    }
  }

  if (isKnownPlaintext(tensor)) {
    FailureOr<Value> gathered = createGather(op, tensor, flatIndex, active);
    if (failed(gathered)) return failure();
    setVector(op.getResult(), *gathered, active);
    return success();
  }

  FailureOr<Value> moved =
      relayout(flatten(tensor), flatIndex, numSlots, slotAt(active), active);
  if (failed(moved)) {
    LDBG() << "cannot align extraction with the slots: " << op;
    return failure();
  }
  setVector(op.getResult(), *moved, active);
  return success();
}

LogicalResult NestVectorizer::processElementwise(Operation* op) {
  bool anyVector = llvm::any_of(op->getOperands(), [&](Value operand) {
    return vectors.contains(operand);
  });
  if (!anyVector) {
    // Materialized lazily if a vectorized op uses it.
    return success();
  }

  DimSet active = activeDimsAt(op);
  SmallVector<Value> operands;
  for (Value operand : op->getOperands()) {
    FailureOr<Value> vector = getVector(operand, active);
    if (failed(vector)) return failure();
    operands.push_back(*vector);
  }
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(operands);
  for (Type type : op->getResultTypes())
    state.addTypes(RankedTensorType::get({numSlots}, type));
  state.addAttributes(op->getAttrs());
  Operation* newOp = builder.create(state);
  bool plaintext = llvm::all_of(operands, [&](Value operand) {
    return plaintextVectors.contains(operand);
  });
  bool uniform = llvm::all_of(operands, [&](Value operand) {
    return uniformVectors.contains(operand);
  });
  for (auto [oldResult, newResult] :
       llvm::zip(op->getResults(), newOp->getResults())) {
    setVector(oldResult, newResult, active);
    if (plaintext) plaintextVectors.insert(newResult);
    if (uniform) uniformVectors.insert(newResult);
  }
  return success();
}

LogicalResult NestVectorizer::processMapLoop(LoopLikeOpInterface loop) {
  // The carried tensor is only used by the nested loop or the final insert.
  return processBlock(loop.getLoopRegions()[0]->front());
}

// A vectorized loop that reduces into a scalar: its iterations are spread over
// slots, so the reduction is over these slots.
LogicalResult NestVectorizer::processReductionLoop(LoopLikeOpInterface loop) {
  unsigned dim = getDim(loop);
  DimSet outer = activeDimsAt(loop);
  DimSet inner = outer | dimBit(dim);
  BlockArgument acc = loop.getRegionIterArgs()[0];
  FailureOr<OperationName> name = getReductionKind(acc);
  if (failed(name)) return failure();

  Type elementType = acc.getType();
  Attribute identity = isMulReduction(*name) ? builder.getOneAttr(elementType)
                                             : builder.getZeroAttr(elementType);
  Value identityVector = arith::ConstantOp::create(
      builder, loop.getLoc(),
      cast<TypedAttr>(DenseElementsAttr::get(
          RankedTensorType::get({numSlots}, elementType), identity)));
  uniformVectors.insert(identityVector);
  plaintextVectors.insert(identityVector);
  setVector(acc, identityVector, inner);
  reductionArgs.insert({acc, *name});

  if (failed(processBlock(loop.getLoopRegions()[0]->front()))) return failure();

  FailureOr<Value> term = getVector(loop.getYieldedValues()[0], inner);
  if (failed(term)) return failure();
  FailureOr<Value> reduced = reduceOverLoop(*term, dim, *name);
  if (failed(reduced)) return failure();
  Value result = *reduced;

  // The root's initial value is combined with the scalar result at the end,
  // and an enclosing loop's iter arg stands for the identity.
  Value init = loop.getInits()[0];
  if (loop != root && !reductionArgs.contains(init) &&
      !isReductionIdentity(init, *name)) {
    FailureOr<Value> initVector = getVector(init, outer);
    if (failed(initVector)) return failure();
    result = createReduction(*name, *initVector, result);
  }
  setVector(loop->getResult(0), result, outer);
  return success();
}

LogicalResult NestVectorizer::processKeptLoop(LoopLikeOpInterface loop) {
  for (BlockArgument iterArg : loop.getRegionIterArgs())
    if (isa<ShapedType>(iterArg.getType())) return failure();

  DimSet active = activeDimsAt(loop);
  SmallVector<Value> inits;
  for (Value init : loop.getInits()) {
    FailureOr<Value> vector = getVector(init, active);
    if (failed(vector)) return failure();
    inits.push_back(*vector);
  }

  Location loc = loop.getLoc();
  LoopLikeOpInterface newLoop;
  if (auto forOp = dyn_cast<affine::AffineForOp>(loop.getOperation())) {
    if (!forOp.hasConstantBounds()) return failure();
    newLoop = affine::AffineForOp::create(
        builder, loc, forOp.getConstantLowerBound(),
        forOp.getConstantUpperBound(), forOp.getStepAsInt(), inits);
  } else {
    auto scfFor = cast<scf::ForOp>(loop.getOperation());
    FailureOr<Value> lb = getScalar(scfFor.getLowerBound());
    FailureOr<Value> ub = getScalar(scfFor.getUpperBound());
    FailureOr<Value> step = getScalar(scfFor.getStep());
    if (failed(lb) || failed(ub) || failed(step)) return failure();
    newLoop = scf::ForOp::create(builder, loc, *lb, *ub, *step, inits);
  }
  keptLoops.push_back(newLoop);

  ivMapping[*loop.getSingleInductionVar()] = *newLoop.getSingleInductionVar();
  for (auto [oldArg, newArg] :
       llvm::zip(loop.getRegionIterArgs(), newLoop.getRegionIterArgs()))
    setVector(oldArg, newArg, active);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&newLoop.getLoopRegions()[0]->front());
    if (failed(processBlock(loop.getLoopRegions()[0]->front())))
      return failure();
    SmallVector<Value> yielded;
    for (Value value : loop.getYieldedValues()) {
      FailureOr<Value> vector = getVector(value, active);
      if (failed(vector)) return failure();
      yielded.push_back(*vector);
    }
    if (isa<affine::AffineForOp>(newLoop.getOperation()))
      affine::AffineYieldOp::create(builder, loc, yielded);
    else
      scf::YieldOp::create(builder, loc, yielded);
  }

  for (auto [oldResult, newResult] :
       llvm::zip(loop->getResults(), newLoop->getResults()))
    setVector(oldResult, newResult, active);
  return success();
}

LogicalResult NestVectorizer::processOp(Operation* op) {
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    if (llvm::is_contained(mapLoops, loop)) return processMapLoop(loop);
    if (isVectorizedLoop(loop)) return processReductionLoop(loop);
    return processKeptLoop(loop);
  }
  if (mapInsert && op == mapInsert.getOperation()) return success();
  if (auto extract = dyn_cast<tensor::ExtractOp>(op))
    return processExtract(extract);
  // Accumulating into the iter arg of a vectorized loop, which stands for the
  // identity of the reduction
  for (OpOperand& operand : op->getOpOperands()) {
    if (!reductionArgs.contains(operand.get())) continue;
    Value term = op->getOperand(1 - operand.getOperandNumber());
    DimSet active = activeDimsAt(op);
    FailureOr<Value> vector = getVector(term, active);
    if (failed(vector)) return failure();
    setVector(op->getResult(0), *vector, active);
    return success();
  }
  // Index computations, constants and slices are materialized on demand.
  if (isa<tensor::ExtractSliceOp, arith::ConstantOp>(op)) return success();
  if (op->getNumResults() > 0 && isMemoryEffectFree(op) &&
      llvm::all_of(op->getResultTypes(),
                   [](Type type) { return type.isIndex(); }))
    return success();
  if (op->hasTrait<OpTrait::Elementwise>() && op->getNumRegions() == 0 &&
      isMemoryEffectFree(op) &&
      llvm::none_of(op->getResultTypes(),
                    [](Type type) { return isa<ShapedType>(type); }))
    return processElementwise(op);
  LDBG() << "unsupported op in loop nest: " << *op;
  return failure();
}

LogicalResult NestVectorizer::processBlock(Block& block) {
  for (Operation& op : block.without_terminator())
    if (failed(processOp(&op))) return failure();
  return success();
}

LogicalResult NestVectorizer::finalizeMapNest() {
  DimSet mapDims = activeDimsAt(mapInsert);
  FailureOr<Value> vector = getVector(mapInsert.getScalar(), mapDims);
  if (failed(vector)) return failure();

  FailureOr<SlotMapInfo> info =
      analyzeSlotMap(mapLoops, renumber(outputIndex, mapLoops), numOutputSlots);
  if (failed(info)) return failure();

  Value result = *vector;
  if (slotSource) {
    FailureOr<Value> moved =
        relayout(result, slotAt(mapDims), numOutputSlots, outputIndex, mapDims);
    if (failed(moved)) return failure();
    result = *moved;
  }

  Value init = root.getInits()[0];
  auto outputType = cast<RankedTensorType>(init.getType());
  Location loc = root.getLoc();
  if (!info->full) {
    // Keep the original values in slots that are not written.
    Value flatInit = flatten(init);
    Type elementType = outputType.getElementType();
    Value mask = createMask(elementType, numOutputSlots, info->image,
                            /*inverted=*/false);
    Value inverseMask = createMask(elementType, numOutputSlots, info->image,
                                   /*inverted=*/true);
    result = createArith(
        /*multiply=*/false,
        createArith(/*multiply=*/true, flatInit, inverseMask),
        createArith(/*multiply=*/true, result, mask));
  }

  if (outputType.getRank() != 1) {
    ReassociationIndices allDims =
        llvm::to_vector(llvm::seq<int64_t>(0, outputType.getRank()));
    result =
        tensor::ExpandShapeOp::create(builder, loc, outputType, result,
                                      ArrayRef<ReassociationIndices>{allDims});
  }
  root->getResult(0).replaceAllUsesWith(result);
  return success();
}

LogicalResult NestVectorizer::finalizeReductionNest() {
  Value reduced = vectors.lookup(root->getResult(0));
  if (!reduced) return failure();
  auto slot = dyn_cast<AffineConstantExpr>(slotAt(0));
  if (!slot) return failure();

  Location loc = root.getLoc();
  Value result =
      tensor::ExtractOp::create(builder, loc, reduced,
                                ValueRange{arith::ConstantIndexOp::create(
                                    builder, loc, slot.getValue())});
  Value init = root.getInits()[0];
  OperationName name = reductionArgs.find(root.getRegionIterArgs()[0])->second;
  if (!isReductionIdentity(init, name))
    result = createReduction(name, init, result);
  root->getResult(0).replaceAllUsesWith(result);
  return success();
}

bool isZeroAttr(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  return false;
}

// Shrink the bounds of a kept loop of the form
//
//   for %i = lb to ub iter_args(%acc = %init) {
//     %w = tensor.extract %table[f(%i, ...)]
//     yield %acc + %x * splat(%w)
//   }
//
// where %table is a constant, to the range of iterations that read a nonzero
// weight for some iteration of the enclosing loops. The other iterations add
// zero, assuming x * 0 = 0, which holds for the finite values FHE programs
// compute on (but not for a float inf or NaN). Unrolling used to remove such
// terms by constant folding.
LogicalResult trimZeroWeightIterations(LoopLikeOpInterface loopLike) {
  auto loop = dyn_cast<affine::AffineForOp>(loopLike.getOperation());
  if (!loop || !loop.hasConstantBounds() || loop.getNumRegionIterArgs() != 1)
    return failure();
  BlockArgument acc = loop.getRegionIterArgs()[0];
  Operation* add = loop.getYieldedValues()[0].getDefiningOp();
  if (!add || !isa<arith::AddIOp, arith::AddFOp>(add)) return failure();
  Value term =
      add->getOperand(0) == acc ? add->getOperand(1) : add->getOperand(0);
  Operation* mul = term.getDefiningOp();
  if (term == acc || !mul || !isa<arith::MulIOp, arith::MulFOp>(mul))
    return failure();
  tensor::ExtractOp weight;
  for (Value operand : mul->getOperands())
    if (auto splat = operand.getDefiningOp<tensor::SplatOp>())
      if (auto extract = splat.getInput().getDefiningOp<tensor::ExtractOp>())
        weight = extract;
  DenseElementsAttr table;
  if (!weight || !matchPattern(weight.getTensor(), m_Constant(&table)))
    return failure();

  SmallVector<LoopLikeOpInterface> loops = {loopLike};
  SmallVector<EnclosingAffineIf> affineIfs;
  if (failed(getEnclosingLoopsAndIfs(loop, loops, affineIfs)) ||
      !affineIfs.empty())
    return failure();
  SmallVector<Value> ivs;
  for (LoopLikeOpInterface l : loops) {
    std::optional<Value> iv = l.getSingleInductionVar();
    if (!iv) return failure();
    ivs.push_back(*iv);
  }
  SmallVector<AffineExpr> indexExprs;
  for (Value index : weight.getIndices()) {
    FailureOr<AffineExpr> expr = getAffineExprOfIVs(index, ivs);
    if (failed(expr)) return failure();
    indexExprs.push_back(*expr);
  }

  // Enumerate the (small) domain of the IVs the weight depends on.
  SmallVector<unsigned> dims = {0};
  for (unsigned i = 1; i < ivs.size(); ++i)
    if (llvm::any_of(indexExprs,
                     [&](AffineExpr e) { return e.isFunctionOfDim(i); }))
      dims.push_back(i);
  SmallVector<ConstantLoopBounds> bounds;
  for (unsigned dim : dims) {
    FailureOr<ConstantLoopBounds> b = getConstantLoopBounds(loops[dim]);
    if (failed(b) || b->tripCount() == 0) return failure();
    bounds.push_back(*b);
  }

  auto tableType = cast<RankedTensorType>(table.getType());
  auto values = table.getValues<Attribute>();
  std::set<int64_t> live;
  bool inBounds = true;
  LogicalResult enumerated = forEachPoint(
      dims, bounds, [&](ArrayRef<std::pair<unsigned, int64_t>> point) {
        int64_t flat = 0;
        for (auto [expr, dimSize] :
             llvm::zip(indexExprs, tableType.getShape())) {
          auto cst = dyn_cast<AffineConstantExpr>(
              substituteDims(expr, ivs.size(), point));
          if (!cst || cst.getValue() < 0 || cst.getValue() >= dimSize) {
            inBounds = false;
            return;
          }
          flat = flat * dimSize + cst.getValue();
        }
        if (!isZeroAttr(values[flat])) live.insert(point[0].second);
      });
  if (failed(enumerated) || !inBounds || live.empty()) return failure();
  int64_t lb = *live.begin();
  int64_t ub = *live.rbegin() + 1;
  int64_t step = 0;
  for (int64_t value : live) step = std::gcd(step, value - lb);
  if (step == 0) step = 1;
  if (lb == loop.getConstantLowerBound() &&
      ub == loop.getConstantUpperBound() && step == loop.getStepAsInt())
    return failure();
  loop.setConstantLowerBound(lb);
  loop.setConstantUpperBound(ub);
  loop.setStep(step);
  return success();
}

// Materialize `expr`, an affine function of `ivs` that does not depend on
// ivs[0], the IV of the loop being replaced.
Value materializeOutsideLoop(OpBuilder& builder, Location loc, AffineExpr expr,
                             ArrayRef<Value> ivs) {
  SmallVector<Value> dimValues(ivs);
  dimValues[0] = Value();
  return affine::expandAffineExpr(builder, loc, expr, dimValues, {});
}

// The weights `weight` of the iterations of `loop`, ordered by increasing
// rotation, as a [steps x numElements] tensor built before `loop`, where `ivs`
// are the IVs of `loop` and its enclosing loops and the weights may not need
// to be rotated by a nonzero `baseIsZero`. Two kinds of weights are supported:
//
// - a row of a plaintext table built by the vectorizer, selected by the IV of
//   `loop`, in which case the table holds the weights of all iterations, and
// - a splat of an element of a constant tensor, in which case the weights of
//   all iterations (and of the enclosing loops the element depends on) are
//   tabulated at compile time.
FailureOr<Value> getWeightTable(OpBuilder& builder, LoopLikeOpInterface loop,
                                Value weight, ArrayRef<Value> ivs,
                                const ConstantLoopBounds& bounds, bool reversed,
                                bool baseIsZero) {
  MLIRContext* ctx = loop->getContext();
  Location loc = loop.getLoc();
  auto weightType = cast<RankedTensorType>(weight.getType());
  int64_t numElements = weightType.getNumElements();
  int64_t steps = bounds.tripCount();
  unsigned numDims = ivs.size();
  AffineExpr row = (getAffineDimExpr(0, ctx) - bounds.lb).floorDiv(bounds.step);
  auto materialize = [&](AffineExpr expr) {
    return materializeOutsideLoop(builder, loc, expr, ivs);
  };

  if (auto slice = weight.getDefiningOp<tensor::ExtractSliceOp>()) {
    // The table rows are not rotated, so the rotations must start at zero, and
    // rows must be in the order of the iterations.
    if (!baseIsZero || reversed ||
        !loop.isDefinedOutsideOfLoop(slice.getSource()) ||
        !slice.hasUnitStride())
      return failure();
    auto sourceType = slice.getSourceType();
    unsigned rank = sourceType.getRank();
    std::optional<int64_t> lastOffset =
        getConstantIntValue(slice.getMixedOffsets().back());
    if (!sourceType.hasStaticShape() ||
        sourceType.getDimSize(rank - 1) != numElements || !lastOffset ||
        *lastOffset != 0)
      return failure();
    // The offsets of the slice as affine functions of `ivs`: one is the row
    // of the iteration, the others do not depend on it.
    std::optional<unsigned> rowDim;
    SmallVector<AffineExpr> offsetExprs;
    for (auto [i, offset] : llvm::enumerate(slice.getMixedOffsets())) {
      if (i == rank - 1) break;
      std::optional<int64_t> size =
          getConstantIntValue(slice.getMixedSizes()[i]);
      if (!size || *size != 1) return failure();
      AffineExpr expr;
      if (std::optional<int64_t> cst = getConstantIntValue(offset)) {
        expr = getAffineConstantExpr(*cst, ctx);
      } else {
        FailureOr<AffineExpr> e = getAffineExprOfIVs(cast<Value>(offset), ivs);
        if (failed(e)) return failure();
        expr = simplifyAffineExpr(*e, numDims, 0);
      }
      if (expr.isFunctionOfDim(0)) {
        if (rowDim || expr != simplifyAffineExpr(row, numDims, 0) ||
            sourceType.getDimSize(i) < steps)
          return failure();
        rowDim = i;
      }
      offsetExprs.push_back(expr);
    }
    if (!rowDim) return failure();

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    for (auto [i, expr] : llvm::enumerate(offsetExprs)) {
      bool isRow = i == *rowDim;
      offsets.push_back(isRow ? OpFoldResult(builder.getIndexAttr(0))
                              : OpFoldResult(materialize(expr)));
      sizes.push_back(builder.getIndexAttr(isRow ? steps : 1));
    }
    offsets.push_back(builder.getIndexAttr(0));
    sizes.push_back(builder.getIndexAttr(numElements));
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
    return tensor::ExtractSliceOp::create(
               builder, loc,
               RankedTensorType::get({steps, numElements},
                                     weightType.getElementType()),
               slice.getSource(), offsets, sizes, strides)
        .getResult();
  }

  // Rotating a splat is the identity, so these weights need no rotation.
  auto splat = weight.getDefiningOp<tensor::SplatOp>();
  auto extract =
      splat ? splat.getInput().getDefiningOp<tensor::ExtractOp>() : nullptr;
  DenseElementsAttr values;
  if (!extract || !matchPattern(extract.getTensor(), m_Constant(&values)))
    return failure();
  SmallVector<AffineExpr> indices;
  for (Value index : extract.getIndices()) {
    FailureOr<AffineExpr> expr = getAffineExprOfIVs(index, ivs);
    if (failed(expr)) return failure();
    indices.push_back(*expr);
  }
  // Table dimensions: the enclosing loops the element depends on, then `loop`
  SmallVector<unsigned> dims;
  SmallVector<ConstantLoopBounds> dimBounds;
  for (unsigned d = 1; d < numDims; ++d) {
    if (llvm::none_of(indices,
                      [&](AffineExpr e) { return e.isFunctionOfDim(d); }))
      continue;
    auto iv = dyn_cast<BlockArgument>(ivs[d]);
    auto outer =
        iv ? dyn_cast<LoopLikeOpInterface>(iv.getOwner()->getParentOp())
           : nullptr;
    if (!outer) return failure();
    FailureOr<ConstantLoopBounds> b = getConstantLoopBounds(outer);
    if (failed(b)) return failure();
    dims.push_back(d);
    dimBounds.push_back(*b);
  }
  dims.push_back(0);
  dimBounds.push_back(bounds);
  SmallVector<int64_t> shape;
  for (const ConstantLoopBounds& b : dimBounds) shape.push_back(b.tripCount());
  shape.push_back(numElements);
  int64_t tableSize = std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                      std::multiplies<int64_t>());
  if (tableSize > kMaxPoints) return failure();

  auto valuesType = cast<RankedTensorType>(values.getType());
  auto sourceValues = values.getValues<Attribute>();
  SmallVector<Attribute> entries(tableSize);
  bool inBounds = true;
  LogicalResult enumerated = forEachPoint(
      dims, dimBounds, [&](ArrayRef<std::pair<unsigned, int64_t>> point) {
        SmallVector<int64_t> sourceIndex;
        for (auto [expr, size] : llvm::zip(indices, valuesType.getShape())) {
          auto cst = dyn_cast<AffineConstantExpr>(
              substituteDims(expr, numDims, point));
          if (!cst || cst.getValue() < 0 || cst.getValue() >= size) {
            inBounds = false;
            return;
          }
          sourceIndex.push_back(cst.getValue());
        }
        SmallVector<int64_t> index;
        for (auto [p, b] : llvm::zip(point, dimBounds))
          index.push_back((p.second - b.lb) / b.step);
        if (reversed) index.back() = steps - 1 - index.back();
        index.push_back(0);
        int64_t start = linearize(index, shape);
        Attribute value =
            sourceValues[linearize(sourceIndex, valuesType.getShape())];
        for (int64_t s = 0; s < numElements; ++s) entries[start + s] = value;
      });
  if (failed(enumerated) || !inBounds) return failure();
  Value table = arith::ConstantOp::create(
      builder, loc,
      cast<TypedAttr>(DenseElementsAttr::get(
          RankedTensorType::get(shape, weightType.getElementType()), entries)));
  if (dims.size() == 1) return table;

  // The weights of the current iteration of the enclosing loops
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  for (auto [d, b] : llvm::zip(ArrayRef(dims).drop_back(), dimBounds)) {
    offsets.push_back(
        materialize((getAffineDimExpr(d, ctx) - b.lb).floorDiv(b.step)));
    sizes.push_back(builder.getIndexAttr(1));
  }
  offsets.append({builder.getIndexAttr(0), builder.getIndexAttr(0)});
  sizes.append(
      {builder.getIndexAttr(steps), builder.getIndexAttr(numElements)});
  SmallVector<OpFoldResult> strides(shape.size(), builder.getIndexAttr(1));
  return tensor::ExtractSliceOp::create(
             builder, loc,
             RankedTensorType::get({steps, numElements},
                                   weightType.getElementType()),
             table, offsets, sizes, strides)
      .getResult();
}

// Replace a kept loop of the form
//
//   for %i = lb to ub step s iter_args(%acc = %init) {
//     %r = tensor_ext.rotate %v, a * %i + b
//     yield %acc + %r
//   }
//
// where %v is loop-invariant, by a rotate_and_reduce of %v. A sum of the
// rotations multiplied by a per-iteration plaintext weight becomes a
// rotate_and_reduce with the weights of all iterations as plaintexts, if they
// form a table indexed by the iteration (see getWeightTable).
LogicalResult foldRotateAccumulateLoop(
    LoopLikeOpInterface loop, llvm::function_ref<bool(Value)> isPlaintext) {
  FailureOr<ConstantLoopBounds> bounds = getConstantLoopBounds(loop);
  if (failed(bounds) || bounds->tripCount() < 1 ||
      loop.getRegionIterArgs().size() != 1)
    return failure();
  BlockArgument acc = loop.getRegionIterArgs()[0];
  auto type = dyn_cast<RankedTensorType>(acc.getType());
  if (!type || type.getRank() != 1 || !acc.hasOneUse()) return failure();

  Value yielded = loop.getYieldedValues()[0];
  Operation* combine = yielded.getDefiningOp();
  if (!combine || !isReduction(combine) ||
      combine->getBlock() != acc.getOwner())
    return failure();
  Value other = combine->getOperand(0) == acc ? combine->getOperand(1)
                                              : combine->getOperand(0);
  if (other == acc) return failure();
  auto rotate = other.getDefiningOp<RotateOp>();
  Value weight;
  if (!rotate) {
    Operation* mul = other.getDefiningOp();
    if (!mul || !isa<arith::MulIOp, arith::MulFOp>(mul) ||
        !isa<arith::AddIOp, arith::AddFOp>(combine) ||
        mul->getBlock() != acc.getOwner())
      return failure();
    for (unsigned i : {0, 1}) {
      if (auto r = mul->getOperand(i).getDefiningOp<RotateOp>()) {
        rotate = r;
        weight = mul->getOperand(1 - i);
        break;
      }
    }
  }
  if (!rotate || !loop.isDefinedOutsideOfLoop(rotate.getTensor()) ||
      (weight && !isPlaintext(weight)))
    return failure();

  SmallVector<LoopLikeOpInterface> enclosing;
  SmallVector<EnclosingAffineIf> affineIfs;
  if (failed(getEnclosingLoopsAndIfs(loop, enclosing, affineIfs)))
    return failure();
  SmallVector<Value> ivs = {*loop.getSingleInductionVar()};
  for (LoopLikeOpInterface outer : enclosing) {
    std::optional<Value> iv = outer.getSingleInductionVar();
    if (!iv) return failure();
    ivs.push_back(*iv);
  }
  FailureOr<AffineExpr> shift = getAffineExprOfIVs(rotate.getShift(), ivs);
  if (failed(shift)) return failure();

  // shift = a * iv + rest, where rest does not depend on iv
  MLIRContext* ctx = loop->getContext();
  unsigned numDims = ivs.size();
  AffineExpr rest = substituteDims(*shift, numDims, {{0, 0}});
  auto coefficient = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(
      substituteDims(*shift, numDims, {{0, 1}}) - rest, numDims, 0));
  if (!coefficient ||
      simplifyAffineExpr(*shift - rest - coefficient * getAffineDimExpr(0, ctx),
                         numDims, 0) != getAffineConstantExpr(0, ctx))
    return failure();

  int64_t numElements = type.getNumElements();
  int64_t steps = bounds->tripCount();
  int64_t period = coefficient.getValue() * bounds->step;
  AffineExpr base = rest + coefficient.getValue() * bounds->lb;
  if (period < 0) {
    base = base + (steps - 1) * period;
    period = -period;
  }
  period %= numElements;
  if (period == 0 || steps > numElements) return failure();

  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  base = simplifyAffineExpr(base, numDims, 0);
  auto materialize = [&](AffineExpr expr) -> Value {
    if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
      int64_t normalized =
          ((cst.getValue() % numElements) + numElements) % numElements;
      return arith::ConstantIndexOp::create(builder, loc, normalized);
    }
    return materializeOutsideLoop(builder, loc, expr, ivs);
  };

  Value plaintexts;
  if (weight) {
    // The baby-step giant-step implementation of a weighted rotate_and_reduce
    // needs about 2 * sqrt(steps) rotations plus one by `base`; keep the loop
    // if its own rotations are fewer.
    auto baseCst = dyn_cast<AffineConstantExpr>(base);
    int64_t loopRotations = steps;
    if (baseCst) {
      loopRotations =
          llvm::count_if(llvm::seq<int64_t>(0, steps), [&](int64_t k) {
            return (baseCst.getValue() + k * period) % numElements != 0;
          });
    }
    int64_t babySteps = static_cast<int64_t>(std::ceil(std::sqrt(steps)));
    int64_t giantSteps = (steps + babySteps - 1) / babySteps;
    int64_t bsgsRotations =
        (babySteps - 1) + (giantSteps - 1) +
        (baseCst && baseCst.getValue() % numElements == 0 ? 0 : 1);
    if (bsgsRotations > loopRotations) return failure();

    FailureOr<Value> table = getWeightTable(
        builder, loop, weight, ivs, *bounds,
        /*reversed=*/coefficient.getValue() < 0,
        /*baseIsZero=*/baseCst && baseCst.getValue() % numElements == 0);
    if (failed(table)) return failure();
    plaintexts = *table;
  }

  // sum_k w_k * rotate(v, base + k * period)
  //   = rotate(sum_k rotate(w_k, -base) * rotate(v, k * period), base)
  auto rotateAndReduce = RotateAndReduceOp::create(
      builder, loc, rotate.getTensor(), plaintexts, period, steps,
      combine->getName().getStringRef());
  Value reduced = rotateAndReduce.getResult();
  auto cst = dyn_cast<AffineConstantExpr>(base);
  if (!cst || cst.getValue() % numElements != 0)
    reduced = RotateOp::create(builder, loc, reduced, materialize(base));

  Value result = reduced;
  if (!isReductionIdentity(loop.getInits()[0], combine->getName())) {
    OperationState state(loc, combine->getName());
    state.addOperands({loop.getInits()[0], reduced});
    state.addTypes(type);
    state.addAttributes(combine->getAttrs());
    result = builder.create(state)->getResult(0);
  }
  loop->getResult(0).replaceAllUsesWith(result);
  loop->erase();
  return convertRotateAndReduceOp(rotateAndReduce);
}

void NestVectorizer::foldKeptLoops() {
  // Inner loops were created after outer loops, so fold in reverse to let a
  // folded inner loop expose an outer one.
  for (LoopLikeOpInterface loop : llvm::reverse(keptLoops)) {
    (void)trimZeroWeightIterations(loop);
    moveLoopInvariantCode(loop);
    (void)foldRotateAccumulateLoop(
        loop, [&](Value value) { return plaintextVectors.contains(value); });
  }
}

void NestVectorizer::rollback() {
  for (Operation* op : llvm::reverse(listener.ops)) {
    op->dropAllUses();
    op->erase();
  }
}

LogicalResult NestVectorizer::run() {
  if (failed(collectLoops())) return failure();
  if (failed(analyzeMapNest())) {
    mapLoops.clear();
    mapInsert = nullptr;
  }
  if (failed(configure())) return failure();

  rootPredecessor = root->getPrevNode();
  if (failed(mapInsert ? processMapLoop(root) : processReductionLoop(root)) ||
      failed(mapInsert ? finalizeMapNest() : finalizeReductionNest())) {
    LDBG() << "failed to vectorize loop nest";
    rollback();
    return failure();
  }

  for (RotateAndReduceOp reduction : pendingReductions)
    (void)convertRotateAndReduceOp(reduction);
  foldKeptLoops();
  root->erase();
  return success();
}

// The candidate slot sources for the nest rooted at `root`, in the order they
// are tried: the insertion into the output of a map nest (as a null
// extraction), then extractions from secret tensors, then from other tensors,
// in program order.
SmallVector<tensor::ExtractOp> getSlotSources(LoopLikeOpInterface root,
                                              const DataFlowSolver* solver) {
  SmallVector<tensor::ExtractOp> secret, other;
  root->walk([&](tensor::ExtractOp op) {
    auto isDuplicate = [&](tensor::ExtractOp existing) {
      return existing.getTensor() == op.getTensor() &&
             llvm::equal(existing.getIndices(), op.getIndices());
    };
    if (llvm::any_of(secret, isDuplicate) || llvm::any_of(other, isDuplicate))
      return;
    (isKnownPlaintext(op.getTensor(), solver) ? other : secret).push_back(op);
  });
  SmallVector<tensor::ExtractOp> sources = {nullptr};
  llvm::append_range(sources, secret);
  llvm::append_range(sources, other);
  // Bound the compile time spent on nests that cannot be vectorized.
  if (sources.size() > 16) sources.resize(16);
  return sources;
}

}  // namespace

struct SimdVectorizeLoops : impl::SimdVectorizeLoopsBase<SimdVectorizeLoops> {
  using SimdVectorizeLoopsBase::SimdVectorizeLoopsBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "failed to run secretness analysis";
      signalPassFailure();
      return;
    }

    SmallVector<LoopLikeOpInterface> roots;
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
        roots.push_back(loop);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });

    for (LoopLikeOpInterface root : roots) {
      bool vectorized = llvm::any_of(
          getSlotSources(root, &solver), [&](tensor::ExtractOp slotSource) {
            return succeeded(NestVectorizer(root, &solver, slotSource).run());
          });
      if (!vectorized) LDBG() << "left loop nest unvectorized: " << root;
    }
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
