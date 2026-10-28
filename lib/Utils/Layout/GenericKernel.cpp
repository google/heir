#include "lib/Utils/Layout/GenericKernel.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/Twine.h"               // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"      // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/IndexingUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"   // from @llvm-project

// ISL
#include "include/isl/ctx.h"         // from @isl
#include "include/isl/map.h"         // from @isl
#include "include/isl/map_type.h"    // from @isl
#include "include/isl/options.h"     // from @isl
#include "include/isl/point.h"       // from @isl
#include "include/isl/set.h"         // from @isl
#include "include/isl/space.h"       // from @isl
#include "include/isl/space_type.h"  // from @isl
#include "include/isl/val.h"         // from @isl

namespace mlir {
namespace heir {

using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

namespace {

// Owns an isl_ctx for the duration of one call.
struct IslContext {
  isl_ctx* ctx;
  IslContext() : ctx(isl_ctx_alloc()) {}
  ~IslContext() { isl_ctx_free(ctx); }
  IslContext(const IslContext&) = delete;
  IslContext& operator=(const IslContext&) = delete;
};

__isl_give isl_map* toIslMap(const IntegerRelation& rel,
                             __isl_keep isl_ctx* ctx) {
  return isl_map_from_basic_map(convertRelationToBasicMap(rel, ctx));
}

// The set { [o0, ..., on] : 0 <= oi < shape[i] }.
__isl_give isl_set* getBoxSet(ArrayRef<int64_t> shape,
                              __isl_keep isl_ctx* ctx) {
  std::string vars, bounds;
  for (auto [i, dim] : llvm::enumerate(shape)) {
    vars += llvm::formatv("{0}o{1}", i ? ", " : "", i).str();
    bounds +=
        llvm::formatv("{0}0 <= o{1} <= {2}", i ? " and " : " : ", i, dim - 1)
            .str();
  }
  std::string str = "{ [" + vars + "]" + bounds + " }";
  return isl_set_read_from_str(ctx, str.c_str());
}

// Checks that a layout maps the operand's data indices to (ct, slot).
LogicalResult validateLayout(const StructuredOperand& operand, StringRef name,
                             std::string& reason) {
  if (operand.layout.getNumRangeVars() != 2) {
    reason = llvm::formatv("layout of {0} must map to (ct, slot)", name);
    return failure();
  }
  if (operand.indexingMap.getNumResults() !=
      operand.layout.getNumDomainVars()) {
    reason = llvm::formatv(
        "indexing map of {0} has {1} results, but its layout has rank {2}",
        name, operand.indexingMap.getNumResults(),
        operand.layout.getNumDomainVars());
    return failure();
  }
  return success();
}

// The map from loop indices to the (ct, slot) positions an operand is read or
// written at: the indexing map composed with the layout.
FailureOr<isl_map*> getAccessMap(const StructuredOperand& operand,
                                 ArrayRef<int64_t> loopRanges,
                                 __isl_keep isl_ctx* ctx, StringRef name,
                                 std::string& reason) {
  if (failed(validateLayout(operand, name, reason))) return failure();
  FailureOr<IntegerRelation> indexRel =
      getIndexingMapRelation(operand.indexingMap, loopRanges);
  if (failed(indexRel)) {
    reason = llvm::formatv("indexing map of {0} is not affine", name);
    return failure();
  }
  isl_map* access = isl_map_apply_range(toIslMap(*indexRel, ctx),
                                        toIslMap(operand.layout, ctx));
  if (!access) {
    reason =
        llvm::formatv("failed to compose indexing map and layout of {0}", name);
    return failure();
  }
  return access;
}

int64_t positiveMod(int64_t lhs, int64_t rhs) {
  return (lhs % rhs + rhs) % rhs;
}

// Evaluates a (symbol-free) affine expression at a point.
int64_t evaluate(AffineExpr expr, ArrayRef<int64_t> dims) {
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) return dims[dim.getPosition()];
  if (auto constant = dyn_cast<AffineConstantExpr>(expr))
    return constant.getValue();
  auto binary = cast<AffineBinaryOpExpr>(expr);
  int64_t lhs = evaluate(binary.getLHS(), dims);
  int64_t rhs = evaluate(binary.getRHS(), dims);
  switch (expr.getKind()) {
    case AffineExprKind::Add:
      return lhs + rhs;
    case AffineExprKind::Mul:
      return lhs * rhs;
    case AffineExprKind::Mod:
      return positiveMod(lhs, rhs);
    case AffineExprKind::FloorDiv:
      return llvm::divideFloorSigned(lhs, rhs);
    case AffineExprKind::CeilDiv:
      return llvm::divideCeilSigned(lhs, rhs);
    default:
      llvm_unreachable("unexpected affine expression");
  }
}

// Converts `map`, from the elements of a tensor of `shape` to (ct, slot), to a
// layout relation. The map must place every element, no two in one slot, as a
// single basic map. Consumes `map` (may be null) and `ctx`.
FailureOr<IntegerRelation> toLayoutRelation(isl_ctx* ctx,
                                            __isl_take isl_map* map,
                                            ArrayRef<int64_t> shape,
                                            StringRef name,
                                            std::string& reason) {
  auto fail = [&](const Twine& why) -> FailureOr<IntegerRelation> {
    reason = why.str();
    isl_map_free(map);
    isl_ctx_free(ctx);
    return failure();
  };
  auto islError = [&]() {
    return fail(isl_ctx_last_error(ctx) == isl_error_quota
                    ? "ISL ran out of operations"
                    : "ISL failed");
  };
  if (!map) return islError();
  isl_set* box = getBoxSet(shape, ctx);
  isl_set* domain = isl_map_domain(isl_map_copy(map));
  isl_bool total = isl_set_is_subset(box, domain);
  isl_set_free(domain);
  isl_set_free(box);
  isl_bool injective =
      total == isl_bool_true ? isl_map_is_injective(map) : isl_bool_true;
  if (total == isl_bool_error || injective == isl_bool_error) return islError();
  if (total != isl_bool_true)
    return fail("the " + name + " relation does not place every element");
  if (injective != isl_bool_true)
    return fail("the " + name + " relation places two elements in one slot");
  if (isl_map_n_basic_map(map) != 1)
    return fail("the " + name + " relation is not a single basic map");
  isl_basic_map* bmap = nullptr;
  isl_map_foreach_basic_map(
      map,
      [](__isl_take isl_basic_map* b, void* user) {
        *static_cast<isl_basic_map**>(user) = b;
        return isl_stat_ok;
      },
      &bmap);
  isl_map_free(map);
  // Frees bmap and ctx.
  return convertBasicMapToRelation(bmap);
}

// The (ct, slot) positions a layout places each data element at.
class SlotTable {
 public:
  static SlotTable get(const presburger::IntegerRelation& layout);

  // The positions of the element at `index`, sorted; none for elements the
  // layout does not place.
  ArrayRef<std::pair<int64_t, int64_t>> lookup(ArrayRef<int64_t> index) const;

  // The lexicographically smallest (ct, (slot - target) mod numSlots) over the
  // positions of the element at `index`: its first ciphertext, and the left
  // rotation that brings a copy in it to slot `target`. (-1, 0) for elements
  // the layout does not place.
  std::pair<int64_t, int64_t> nearest(ArrayRef<int64_t> index, int64_t target,
                                      int64_t numSlots) const;

  // The positions of every element.
  ArrayRef<SmallVector<std::pair<int64_t, int64_t>, 1>> positions() const {
    return slots;
  }

 private:
  int64_t flatten(ArrayRef<int64_t> index) const;

  SmallVector<int64_t> shape;
  std::vector<SmallVector<std::pair<int64_t, int64_t>, 1>> slots;
};

}  // namespace

SlotTable SlotTable::get(const IntegerRelation& layout) {
  // Enumerates the image of each data index separately, which avoids a
  // parametric solve over the layout's mod/floor variables.
  unsigned rank = layout.getNumDomainVars();
  PointPairCollector collector(rank, 2);
  enumeratePoints(layout, collector);
  SlotTable table;
  table.shape.assign(rank, 0);
  for (const auto& [index, position] : collector.points) {
    for (unsigned i = 0; i < rank; ++i) {
      table.shape[i] = std::max(table.shape[i], index[i] + 1);
    }
  }
  int64_t size = 1;
  for (int64_t dim : table.shape) size *= dim;
  table.slots.resize(size);
  for (const auto& [index, position] : collector.points) {
    table.slots[table.flatten(index)].push_back({position[0], position[1]});
  }
  for (auto& positions : table.slots) llvm::sort(positions);
  return table;
}

ArrayRef<std::pair<int64_t, int64_t>> SlotTable::lookup(
    ArrayRef<int64_t> index) const {
  for (auto [i, dim] : llvm::zip(index, shape)) {
    if (i < 0 || i >= dim) return {};
  }
  return slots[flatten(index)];
}

std::pair<int64_t, int64_t> SlotTable::nearest(ArrayRef<int64_t> index,
                                               int64_t target,
                                               int64_t numSlots) const {
  ArrayRef<std::pair<int64_t, int64_t>> positions = lookup(index);
  if (positions.empty()) return {-1, 0};
  // The positions in the first ciphertext, sorted by slot.
  int64_t ct = positions.front().first;
  auto end = llvm::find_if(positions, [&](auto p) { return p.first != ct; });
  auto first = positions.begin();
  auto atOrAfter = std::lower_bound(first, end, std::make_pair(ct, target));
  if (atOrAfter != end) return {ct, atOrAfter->second - target};
  return {ct, first->second + numSlots - target};
}

int64_t SlotTable::flatten(ArrayRef<int64_t> index) const {
  int64_t flat = 0;
  for (auto [i, dim] : llvm::zip(index, shape)) flat = flat * dim + i;
  return flat;
}

FailureOr<InstanceGroups> groupInstances(const GenericKernelSpec& spec,
                                         int64_t numCts,
                                         GenericReductionKind kind,
                                         std::string& reason) {
  if (failed(validateGenericKernelSpec(spec, reason))) return failure();
  const int64_t numSlots = spec.numSlots;
  // Where each operand element lives.
  SlotTable outputSlots = SlotTable::get(spec.output.layout);
  std::vector<std::optional<SlotTable>> inputSlots(spec.inputs.size());
  std::vector<SmallVector<int64_t>> strides(spec.inputs.size());
  for (auto [i, input] : llvm::enumerate(spec.inputs)) {
    if (input.isCleartext()) {
      strides[i] = computeStrides(*input.cleartextShape);
      continue;
    }
    inputSlots[i] = SlotTable::get(input.layout);
  }

  // Walk the loop nest. Each (iteration, output slot) instance has a key
  // [ct_o, (ct_i, shift_i) per secret input], taking the lexicographically
  // smallest (ct_i, shift_i) of a replicated input element and ct_i = -1 for
  // one the layout does not place (read as zero). Per key and output slot,
  // collect the distinct cleartext reads (row-major indices) with their
  // multiplicities.
  InstanceGroups groups;
  auto& readsByKey = groups.readsByKey;
  auto& coverage = groups.coverage;
  const unsigned numLoops = spec.loopRanges.size();
  SmallVector<int64_t> d(numLoops, 0);
  for (bool done = llvm::is_contained(spec.loopRanges, 0); !done;) {
    SmallVector<int64_t> cleartextReads;
    std::vector<SmallVector<int64_t>> inputIndices(spec.inputs.size());
    for (auto [i, input] : llvm::enumerate(spec.inputs)) {
      inputIndices[i] = evaluateAffineMap(input.indexingMap, d);
      if (input.isCleartext())
        cleartextReads.push_back(linearize(inputIndices[i], strides[i]));
    }
    for (auto [outputCt, outputSlot] :
         outputSlots.lookup(evaluateAffineMap(spec.output.indexingMap, d))) {
      std::vector<int64_t> key = {outputCt};
      for (auto [i, input] : llvm::enumerate(spec.inputs)) {
        if (input.isCleartext()) continue;
        auto [ct, shift] =
            inputSlots[i]->nearest(inputIndices[i], outputSlot, numSlots);
        key.push_back(ct);
        key.push_back(shift);
      }
      InstanceGroups::Reads& reads = readsByKey[key][outputSlot];
      auto it = llvm::find_if(reads, [&](const auto& read) {
        return read.first == cleartextReads;
      });
      if (it == reads.end()) {
        reads.push_back({cleartextReads, 1});
      } else {
        ++it->second;
      }
    }
    // Advance the iteration in row-major order.
    int64_t j = static_cast<int64_t>(numLoops) - 1;
    while (j >= 0 && ++d[j] == spec.loopRanges[j]) d[j--] = 0;
    done = j < 0;
  }

  coverage.assign(numCts, std::vector<int64_t>(numSlots, 0));
  for (const auto& [key, slots] : readsByKey) {
    if (key[0] >= numCts) {
      reason =
          "the output layout writes more ciphertexts than the result holds";
      return failure();
    }
    for (const auto& [slot, reads] : slots) {
      for (const auto& [read, count] : reads) {
        coverage[key[0]][slot] += count;
        if (kind == GenericReductionKind::Mul && count > 1) {
          reason =
              "a multiplicative reduction cannot repeat a factor in a slot";
          return failure();
        }
      }
      if (kind == GenericReductionKind::None && coverage[key[0]][slot] > 1) {
        reason =
            "an output slot is written by several iterations of an op "
            "without reductions";
        return failure();
      }
    }
  }
  groups.writesPartially.assign(numCts, false);
  for (const auto& positions : outputSlots.positions()) {
    for (auto [ct, slot] : positions) {
      if (ct < numCts && coverage[ct][slot] == 0)
        groups.writesPartially[ct] = true;
    }
  }
  return groups;
}

namespace {

// Counts the points of `set` (taken), up to `limit`; nullopt beyond it.
std::optional<int64_t> countPoints(__isl_take isl_set* set, int64_t limit) {
  struct Counter {
    int64_t count = 0;
    int64_t limit;
  } counter{0, limit};
  isl_stat status = isl_set_foreach_point(
      set,
      [](__isl_take isl_point* point, void* user) {
        isl_point_free(point);
        auto* counter = static_cast<Counter*>(user);
        return ++counter->count > counter->limit ? isl_stat_error : isl_stat_ok;
      },
      &counter);
  isl_set_free(set);
  if (status != isl_stat_ok) return std::nullopt;
  return counter.count;
}

}  // namespace

FailureOr<GroupCount> countInstanceGroups(const GenericKernelSpec& spec,
                                          int64_t maxKeys,
                                          unsigned long maxOperations,
                                          std::string& reason) {
  if (failed(validateGenericKernelSpec(spec, reason))) return failure();
  IslContext isl;
  isl_options_set_on_error(isl.ctx, ISL_ON_ERROR_CONTINUE);
  isl_ctx_set_max_operations(isl.ctx, maxOperations);
  auto fail = [&](StringRef why) -> FailureOr<GroupCount> {
    reason = why.str();
    return failure();
  };

  FailureOr<isl_map*> output =
      getAccessMap(spec.output, spec.loopRanges, isl.ctx, "the output", reason);
  if (failed(output)) return failure();
  // Instances [d -> (ct_o, slot_o)] to their output ciphertext.
  isl_map* key = isl_map_project_out(isl_map_range_map(isl_map_copy(*output)),
                                     isl_dim_out, 1, 1);
  // (ct_o, slot_o, ct_i, slot_i) -> (ct_i, (slot_i - slot_o) mod numSlots), as
  // two pieces: with a quotient variable for the mod, the lexmin below splits
  // on it over the whole loop nest.
  isl_map* shift = isl_map_read_from_str(
      isl.ctx, llvm::formatv("{ [co, so, ci, si] -> [ci, si - so] : si >= so; "
                             "[co, so, ci, si] -> [ci, si - so + {0}] : "
                             "si < so }",
                             spec.numSlots)
                   .str()
                   .c_str());
  int64_t numRotations = 0;
  for (auto [i, input] : llvm::enumerate(spec.inputs)) {
    if (input.isCleartext()) continue;
    FailureOr<IntegerRelation> index =
        getIndexingMapRelation(input.indexingMap, spec.loopRanges);
    if (failed(index)) {
      isl_map_free(key);
      isl_map_free(shift);
      isl_map_free(*output);
      return fail(
          llvm::formatv("indexing map of input {0} is not affine", i).str());
    }
    // The smallest (ct, rotation) over the replicas of an element has the
    // element's first ciphertext, which a lexmin over the layout alone finds;
    // only the replicas in it take part in the lexmin over the instances.
    isl_map* layout = toIslMap(input.layout, isl.ctx);
    isl_map* firstCt = isl_map_project_out(isl_map_lexmin(isl_map_copy(layout)),
                                           isl_dim_out, 1, 1);
    isl_map* anySlot = isl_map_universe(
        isl_space_alloc(isl.ctx, 0, input.layout.getNumDomainVars(), 1));
    layout =
        isl_map_intersect(layout, isl_map_flat_range_product(firstCt, anySlot));
    isl_map* access = isl_map_apply_range(toIslMap(*index, isl.ctx), layout);
    // Instances to the (ct, rotation) of the input element they read, the
    // lexicographically smallest over its replicas.
    isl_map* positions = isl_map_flat_range_product(
        isl_map_range_map(isl_map_copy(*output)),
        isl_map_apply_range(isl_map_domain_map(isl_map_copy(*output)), access));
    isl_map* inputKey =
        isl_map_lexmin(isl_map_apply_range(positions, isl_map_copy(shift)));
    // Elements the layout does not place read as zero, with key (-1, 0).
    isl_set* unplaced =
        isl_set_subtract(isl_map_domain(isl_map_copy(key)),
                         isl_map_domain(isl_map_copy(inputKey)));
    inputKey = isl_map_union(
        inputKey, isl_map_from_domain_and_range(
                      unplaced, isl_set_read_from_str(isl.ctx, "{ [-1, 0] }")));
    std::optional<int64_t> rotations = countPoints(
        isl_set_subtract(isl_map_range(isl_map_copy(inputKey)),
                         isl_set_read_from_str(isl.ctx, "{ [ct, 0] }")),
        maxKeys);
    key = isl_map_flat_range_product(key, inputKey);
    if (!key || !rotations) {
      isl_map_free(key);
      isl_map_free(shift);
      isl_map_free(*output);
      return fail("too many rotations, or ISL ran out of operations");
    }
    numRotations += *rotations;
  }
  isl_map_free(shift);
  isl_map_free(*output);
  std::optional<int64_t> numKeys = countPoints(isl_map_range(key), maxKeys);
  if (!numKeys) return fail("too many groups, or ISL ran out of operations");
  return GroupCount{*numKeys, numRotations};
}

SmallVector<int64_t> evaluateAffineMap(AffineMap map, ArrayRef<int64_t> point) {
  SmallVector<int64_t> result;
  for (AffineExpr expr : map.getResults()) {
    result.push_back(evaluate(expr, point));
  }
  return result;
}

LogicalResult validateGenericKernelSpec(const GenericKernelSpec& spec,
                                        std::string& reason) {
  if (failed(validateLayout(spec.output, "the output", reason))) {
    return failure();
  }
  for (auto [i, input] : llvm::enumerate(spec.inputs)) {
    if (!input.isCleartext() &&
        failed(validateLayout(input, llvm::formatv("input {0}", i).str(),
                              reason))) {
      return failure();
    }
  }
  // Two distinct output elements in one slot would have no slot-wise meaning.
  IslContext isl;
  isl_map* outputLayout = toIslMap(spec.output.layout, isl.ctx);
  isl_bool injective = isl_map_is_injective(outputLayout);
  isl_map_free(outputLayout);
  if (injective != isl_bool_true) {
    reason =
        "the output layout places two distinct output elements in the same "
        "(ct, slot)";
    return failure();
  }
  return success();
}

FailureOr<IntegerRelation> getIndexingMapRelation(
    AffineMap map, ArrayRef<int64_t> loopRanges) {
  assert(map.getNumDims() == loopRanges.size() &&
         "indexing map must have one dimension per loop");
  assert(map.getNumSymbols() == 0 && "indexing maps must not have symbols");
  // getRelationFromMap replaces the space with a set space whose leading
  // variables are the map's dimensions; those are moved into the domain.
  IntegerRelation rel(PresburgerSpace::getRelationSpace());
  if (failed(affine::getRelationFromMap(map, rel))) {
    return failure();
  }
  if (rel.getNumDomainVars() == 0 && map.getNumDims() > 0) {
    rel.convertVarKind(VarKind::Range, 0, map.getNumDims(), VarKind::Domain, 0);
  }
  for (auto [i, range] : llvm::enumerate(loopRanges)) {
    rel.addBound(presburger::BoundType::LB, i, 0);
    rel.addBound(presburger::BoundType::UB, i, range - 1);
  }
  return rel;
}

bool outputMapCoversResult(AffineMap outputMap, ArrayRef<int64_t> loopRanges,
                           ArrayRef<int64_t> outputShape) {
  FailureOr<IntegerRelation> rel =
      getIndexingMapRelation(outputMap, loopRanges);
  if (failed(rel)) return false;
  IslContext isl;
  isl_set* written = isl_map_range(toIslMap(*rel, isl.ctx));
  isl_set* box = getBoxSet(outputShape, isl.ctx);
  isl_bool covers = isl_set_is_subset(box, written);
  isl_set_free(written);
  isl_set_free(box);
  return covers == isl_bool_true;
}

FailureOr<IntegerRelation> inferInducedOutputLayout(
    ArrayRef<int64_t> loopRanges, AffineMap inputMap,
    const IntegerRelation& inputLayout, AffineMap outputMap,
    ArrayRef<int64_t> outputShape, bool reduceToLexmin,
    unsigned long maxOperations, std::string& reason) {
  // convertBasicMapToRelation frees this context on success, so every other
  // ISL object must be released before that call.
  isl_ctx* ctx = isl_ctx_alloc();
  isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
  isl_ctx_set_max_operations(ctx, maxOperations);
  auto fail = [&](std::string why) -> FailureOr<IntegerRelation> {
    reason = std::move(why);
    isl_ctx_free(ctx);
    return failure();
  };

  FailureOr<isl_map*> inputAccess =
      getAccessMap(StructuredOperand{inputMap, inputLayout}, loopRanges, ctx,
                   "input", reason);
  if (failed(inputAccess)) return fail(reason);
  FailureOr<IntegerRelation> outputIndex =
      getIndexingMapRelation(outputMap, loopRanges);
  if (failed(outputIndex)) {
    isl_map_free(*inputAccess);
    return fail("output indexing map is not affine");
  }
  // o -> (ct, slot): every output element lands where the input element read
  // in the same iteration lives.
  isl_map* induced =
      isl_map_apply_domain(*inputAccess, toIslMap(*outputIndex, ctx));
  if (reduceToLexmin) induced = isl_map_lexmin(induced);
  return toLayoutRelation(ctx, isl_map_coalesce(induced), outputShape,
                          "induced", reason);
}

FailureOr<IntegerRelation> pullBackInputLayout(
    ArrayRef<int64_t> loopRanges, ArrayRef<bool> isReduction,
    AffineMap inputMap, ArrayRef<int64_t> inputShape, AffineMap outputMap,
    const IntegerRelation& outputLayout, unsigned long maxOperations,
    std::string& reason) {
  // convertBasicMapToRelation frees this context on success, so every other
  // ISL object must be released before that call.
  isl_ctx* ctx = isl_ctx_alloc();
  isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
  isl_ctx_set_max_operations(ctx, maxOperations);
  auto fail = [&](std::string why) -> FailureOr<IntegerRelation> {
    reason = std::move(why);
    isl_ctx_free(ctx);
    return failure();
  };

  MLIRContext* mlirCtx = inputMap.getContext();
  AffineExpr reductionIndex = getAffineConstantExpr(0, mlirCtx);
  int64_t numReductions = 1;
  for (int64_t j = loopRanges.size() - 1; j >= 0; --j) {
    if (!isReduction[j]) continue;
    reductionIndex =
        reductionIndex + numReductions * getAffineDimExpr(j, mlirCtx);
    numReductions *= loopRanges[j];
  }
  FailureOr<IntegerRelation> reductionRel = getIndexingMapRelation(
      AffineMap::get(loopRanges.size(), 0, reductionIndex), loopRanges);
  FailureOr<IntegerRelation> inputIndex =
      getIndexingMapRelation(inputMap, loopRanges);
  FailureOr<isl_map*> output =
      getAccessMap(StructuredOperand{outputMap, outputLayout}, loopRanges, ctx,
                   "the output", reason);
  if (failed(output)) return fail(reason);
  if (failed(reductionRel) || failed(inputIndex)) {
    isl_map_free(*output);
    return fail("the indexing maps are not affine");
  }
  // d -> (ct_o * R + r, slot_o), then input element -> that position.
  isl_map* positions = isl_map_apply_range(
      isl_map_flat_range_product(*output, toIslMap(*reductionRel, ctx)),
      isl_map_read_from_str(
          ctx,
          llvm::formatv("{ [co, so, r] -> [co * {0} + r, so] }", numReductions)
              .str()
              .c_str()));
  isl_map* pulled = isl_map_coalesce(
      isl_map_apply_domain(positions, toIslMap(*inputIndex, ctx)));
  return toLayoutRelation(ctx, pulled, inputShape, "pulled-back", reason);
}

}  // namespace heir
}  // namespace mlir
