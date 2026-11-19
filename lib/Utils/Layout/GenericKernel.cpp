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
#include "include/isl/aff.h"         // from @isl
#include "include/isl/ctx.h"         // from @isl
#include "include/isl/map.h"         // from @isl
#include "include/isl/map_type.h"    // from @isl
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
  isl_map* access = isl_map_preimage_domain_pw_multi_aff(
      toIslMap(operand.layout, ctx),
      isl_pw_multi_aff_from_map(toIslMap(*indexRel, ctx)));
  if (!access) {
    reason =
        llvm::formatv("failed to compose indexing map and layout of {0}", name);
    return failure();
  }
  return access;
}

// Detecting equalities and removing redundancies turns existential variables
// into explicit divisions where possible; isl_map_lexmin otherwise computes
// them by sampling.
__isl_give isl_map* simplify(__isl_take isl_map* map) {
  return isl_map_coalesce(
      isl_map_remove_redundancies(isl_map_detect_equalities(map)));
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

// The elements `map` accesses in the iterations where `otherMap` accesses an
// element `otherLayout` places. The layout's domain is taken on its own, so its
// local variables stay out of the projection of the loops.
__isl_give isl_set* getElementsWithPlacedPartner(
    const IntegerRelation& map, const IntegerRelation& otherMap,
    const IntegerRelation& otherLayout, __isl_keep isl_ctx* ctx) {
  isl_set* placed = isl_map_domain(toIslMap(otherLayout, ctx));
  isl_set* iterations =
      isl_map_domain(isl_map_intersect_range(toIslMap(otherMap, ctx), placed));
  return isl_set_apply(iterations, toIslMap(map, ctx));
}

// Converts `map`, from the elements of a tensor of `shape` to (ct, slot), to a
// layout relation. The map must place every element, no two in one slot, as a
// single basic map; `domain` is the domain of `map`. Consumes `map` and
// `domain` (either may be null) and `ctx`.
FailureOr<IntegerRelation> toLayoutRelation(
    isl_ctx* ctx, __isl_take isl_map* map, __isl_take isl_set* domain,
    ArrayRef<int64_t> shape, StringRef name, std::string& reason) {
  auto fail = [&](const Twine& why) -> FailureOr<IntegerRelation> {
    reason = why.str();
    isl_map_free(map);
    isl_set_free(domain);
    isl_ctx_free(ctx);
    return failure();
  };
  auto islError = [&]() { return fail("ISL failed"); };
  if (!map || !domain) return islError();
  isl_set* box = getBoxSet(shape, ctx);
  isl_bool total = isl_set_is_subset(box, domain);
  isl_set_free(box);
  isl_set_free(domain);
  domain = nullptr;
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
    ArrayRef<int64_t> outputShape, bool reduceToLexmin, std::string& reason) {
  // convertBasicMapToRelation frees this context on success, so every other
  // ISL object must be released before that call.
  isl_ctx* ctx = isl_ctx_alloc();
  auto fail = [&](std::string why) -> FailureOr<IntegerRelation> {
    reason = std::move(why);
    isl_ctx_free(ctx);
    return failure();
  };

  FailureOr<isl_map*> inputAccess =
      getAccessMap(StructuredOperand{inputMap, inputLayout}, loopRanges, ctx,
                   "input", reason);
  if (failed(inputAccess)) return fail(reason);
  FailureOr<IntegerRelation> inputIndex =
      getIndexingMapRelation(inputMap, loopRanges);
  FailureOr<IntegerRelation> outputIndex =
      getIndexingMapRelation(outputMap, loopRanges);
  if (failed(inputIndex) || failed(outputIndex)) {
    isl_map_free(*inputAccess);
    return fail("the indexing maps are not affine");
  }
  isl_map* induced;
  if (!reduceToLexmin) {
    // o -> (ct, slot): every output element lands where the input element
    // read in the same iteration lives.
    induced = isl_map_apply_domain(*inputAccess, toIslMap(*outputIndex, ctx));
  } else {
    isl_map_free(*inputAccess);
    // The smallest position among those of all elements an output reads is
    // the smallest among their first positions, so only the first replica of
    // each element enters the lexmin. The loops stay in the range, o -> (ct,
    // slot, d), so the lexmin has no existential variables for them; the
    // smallest (ct, slot, d) has the smallest (ct, slot), and projecting out
    // d leaves it.
    // Composing by isl_map_apply_range eliminates the input element and
    // derives the divisions of `access` afresh; a preimage keeps those of the
    // lexmin result, whose large multipliers made the second lexmin run for
    // minutes.
    isl_map* first = isl_map_lexmin(simplify(toIslMap(inputLayout, ctx)));
    isl_map* access = isl_map_apply_range(toIslMap(*inputIndex, ctx), first);
    isl_map* loops = isl_map_identity(
        isl_space_map_from_set(isl_space_domain(isl_map_get_space(access))));
    isl_map* withLoops = isl_map_apply_domain(
        isl_map_flat_range_product(access, loops), toIslMap(*outputIndex, ctx));
    induced = isl_map_project_out(isl_map_lexmin(simplify(withLoops)),
                                  isl_dim_out, 2, loopRanges.size());
  }
  isl_set* domain =
      getElementsWithPlacedPartner(*outputIndex, *inputIndex, inputLayout, ctx);
  return toLayoutRelation(ctx, isl_map_coalesce(induced), domain, outputShape,
                          "induced", reason);
}

namespace {

// The map from the elements a layout places to the flat position
// ct * numSlots + slot of their first copy.
__isl_give isl_map* getFirstFlatPosition(const IntegerRelation& layout,
                                         int64_t numSlots,
                                         __isl_keep isl_ctx* ctx) {
  std::string flat =
      llvm::formatv("{ [ct, slot] -> [ct * {0} + slot] }", numSlots).str();
  return isl_map_lexmin(simplify(isl_map_apply_range(
      toIslMap(layout, ctx), isl_map_read_from_str(ctx, flat.c_str()))));
}

// The smallest or largest value of dim `pos` of a bounded, nonempty set.
int64_t getDimBound(__isl_keep isl_set* set, unsigned pos, bool max) {
  isl_set* dim = isl_set_copy(set);
  unsigned numDims = isl_set_dim(dim, isl_dim_set);
  dim = isl_set_project_out(dim, isl_dim_set, pos + 1, numDims - pos - 1);
  dim = isl_set_project_out(dim, isl_dim_set, 0, pos);
  isl_point* point =
      isl_set_sample_point(max ? isl_set_lexmax(dim) : isl_set_lexmin(dim));
  isl_val* value = isl_point_get_coordinate_val(point, isl_dim_set, 0);
  int64_t result = isl_val_get_num_si(value);
  isl_val_free(value);
  isl_point_free(point);
  return result;
}

// The number of ciphertexts a layout uses.
int64_t getNumCiphertexts(__isl_keep isl_map* layout) {
  isl_set* range = isl_map_range(isl_map_copy(layout));
  int64_t result = getDimBound(range, 0, /*max=*/true) + 1;
  isl_set_free(range);
  return result;
}

// Whether the rotation an input needs, (flat_in(F(d)) - flat_out(G(d))) mod
// numSlots, is a function of the loops in `fixed`: then the input needs no
// rotation that depends on the other loops. Iterations that read an element
// the input layout does not place read zero and do not count.
FailureOr<bool> isAligned(ArrayRef<int64_t> loopRanges,
                          const IntegerRelation& inputIndex,
                          const IntegerRelation& inputLayout,
                          const IntegerRelation& outputIndex,
                          const IntegerRelation& outputLayout,
                          ArrayRef<unsigned> fixed, int64_t numSlots,
                          __isl_keep isl_ctx* ctx) {
  isl_map* input =
      isl_map_apply_range(toIslMap(inputIndex, ctx),
                          getFirstFlatPosition(inputLayout, numSlots, ctx));
  isl_map* output =
      isl_map_apply_range(toIslMap(outputIndex, ctx),
                          getFirstFlatPosition(outputLayout, numSlots, ctx));
  std::string loops, kept;
  for (unsigned j = 0; j < loopRanges.size(); ++j) {
    loops += llvm::formatv("{0}d{1}", j ? ", " : "", j).str();
  }
  for (auto [i, j] : llvm::enumerate(fixed)) {
    kept += llvm::formatv("{0}d{1}", i ? ", " : "", j).str();
  }
  std::string projection =
      llvm::formatv("{ [{0}] -> [{1}] }", loops, kept).str();
  std::string difference =
      llvm::formatv("{ [a, b] -> [(a - b) mod {0}] }", numSlots).str();
  isl_map* shift =
      isl_map_apply_range(isl_map_flat_range_product(input, output),
                          isl_map_read_from_str(ctx, difference.c_str()));
  shift = isl_map_apply_domain(shift,
                               isl_map_read_from_str(ctx, projection.c_str()));
  shift = simplify(shift);
  isl_bool singleValued = isl_map_is_single_valued(shift);
  isl_map_free(shift);
  if (singleValued == isl_bool_error) return failure();
  return singleValued == isl_bool_true;
}

// The linear coefficients of `expr` over `numDims` loops, or failure if it is
// not a linear combination of the loops plus a constant.
FailureOr<SmallVector<int64_t>> getLinearCoefficients(AffineExpr expr,
                                                      unsigned numDims) {
  bool linear = true;
  expr.walk([&](AffineExpr sub) {
    if (sub.getKind() == AffineExprKind::Mod ||
        sub.getKind() == AffineExprKind::FloorDiv ||
        sub.getKind() == AffineExprKind::CeilDiv || isa<AffineSymbolExpr>(sub))
      linear = false;
  });
  if (!linear) return failure();
  SmallVector<int64_t> point(numDims, 0);
  int64_t constant = evaluate(expr, point);
  SmallVector<int64_t> coefficients;
  for (unsigned j = 0; j < numDims; ++j) {
    point[j] = 1;
    coefficients.push_back(evaluate(expr, point) - constant);
    point[j] = 0;
  }
  return coefficients;
}

}  // namespace

int64_t getNumCiphertexts(const IntegerRelation& layout) {
  IslContext isl;
  isl_map* map = toIslMap(layout, isl.ctx);
  int64_t result = getNumCiphertexts(map);
  isl_map_free(map);
  return result;
}

FailureOr<std::optional<IntegerRelation>> getPolyphaseInputLayout(
    ArrayRef<int64_t> loopRanges, AffineMap inputMap,
    const IntegerRelation& inputLayout, AffineMap outputMap,
    const IntegerRelation& outputLayout, ArrayRef<int64_t> outputShape,
    int64_t numSlots, std::string& reason) {
  // convertBasicMapToRelation frees this context on success, so every other
  // ISL object must be released before that call.
  isl_ctx* ctx = isl_ctx_alloc();
  using Result = FailureOr<std::optional<IntegerRelation>>;
  auto fail = [&](const Twine& why) -> Result {
    reason = why.str();
    isl_ctx_free(ctx);
    return failure();
  };
  const unsigned numLoops = loopRanges.size();
  FailureOr<IntegerRelation> inputIndex =
      getIndexingMapRelation(inputMap, loopRanges);
  FailureOr<IntegerRelation> outputIndex =
      getIndexingMapRelation(outputMap, loopRanges);
  if (failed(inputIndex) || failed(outputIndex))
    return fail("the indexing maps are not affine");

  // The output dim each loop writes, for loops the output map reads as a
  // plain dimension; the other loops are reductions, or parallel loops the
  // classification below does not pair.
  SmallVector<int64_t> outputDimOfLoop(numLoops, -1);
  SmallVector<bool> writtenLoop(numLoops, false);
  for (auto [m, expr] : llvm::enumerate(outputMap.getResults())) {
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      outputDimOfLoop[dim.getPosition()] = m;
    }
    for (unsigned j = 0; j < numLoops; ++j) {
      if (expr.isFunctionOfDim(j)) writtenLoop[j] = true;
    }
  }

  // Each input result reads either one written loop with a positive stride
  // (paired with the output dim that loop writes), plus any reduction loops,
  // or reduction loops only (a class result).
  struct Paired {
    unsigned result;
    int64_t stride;
    int64_t outputDim;
  };
  SmallVector<Paired> paired;
  SmallVector<unsigned> classes;
  SmallVector<bool> pairedLoop(numLoops, false);
  for (auto [k, expr] : llvm::enumerate(inputMap.getResults())) {
    FailureOr<SmallVector<int64_t>> coefficients =
        getLinearCoefficients(expr, numLoops);
    if (failed(coefficients)) return fail("an input index is not linear");
    SmallVector<unsigned> written;
    for (unsigned j = 0; j < numLoops; ++j) {
      if ((*coefficients)[j] != 0 && writtenLoop[j]) written.push_back(j);
    }
    if (written.empty()) {
      classes.push_back(k);
      continue;
    }
    unsigned j = written.front();
    if (written.size() > 1 || (*coefficients)[j] <= 0 ||
        outputDimOfLoop[j] < 0 || pairedLoop[j]) {
      return fail(
          "an input index does not read one output loop with a positive "
          "stride");
    }
    pairedLoop[j] = true;
    paired.push_back(
        {static_cast<unsigned>(k), (*coefficients)[j], outputDimOfLoop[j]});
  }
  // The rotation may depend on the loops the input does not pair: the
  // reductions and the output dims it does not read (a convolution's output
  // channel).
  SmallVector<unsigned> fixed;
  for (unsigned j = 0; j < numLoops; ++j) {
    if (!pairedLoop[j]) fixed.push_back(j);
  }

  FailureOr<bool> aligned =
      isAligned(loopRanges, *inputIndex, inputLayout, *outputIndex,
                outputLayout, fixed, numSlots, ctx);
  if (failed(aligned)) return fail("ISL failed");
  if (*aligned) {
    isl_ctx_free(ctx);
    return std::optional<IntegerRelation>();
  }

  // The output pitches: the flat position of the output must be affine in its
  // index, with positive pitches for the paired dims.
  isl_pw_multi_aff* outputFlat = isl_pw_multi_aff_from_map(
      getFirstFlatPosition(outputLayout, numSlots, ctx));
  isl_pw_aff* outputFlatAff = isl_pw_multi_aff_get_pw_aff(outputFlat, 0);
  isl_pw_multi_aff_free(outputFlat);
  isl_aff* outputAff = nullptr;
  if (isl_pw_aff_n_piece(outputFlatAff) == 1) {
    isl_pw_aff_foreach_piece(
        outputFlatAff,
        [](__isl_take isl_set* set, __isl_take isl_aff* aff, void* user) {
          isl_set_free(set);
          *static_cast<isl_aff**>(user) = aff;
          return isl_stat_ok;
        },
        &outputAff);
  }
  isl_pw_aff_free(outputFlatAff);
  if (!outputAff)
    return fail("the output position is not affine in the output index");
  bool hasDivs = isl_aff_dim(outputAff, isl_dim_div) > 0;
  SmallVector<int64_t> pitch;
  for (unsigned m = 0; m < outputShape.size(); ++m) {
    isl_val* v = isl_aff_get_coefficient_val(outputAff, isl_dim_in, m);
    pitch.push_back(isl_val_get_num_si(v));
    isl_val_free(v);
  }
  isl_aff_free(outputAff);
  if (hasDivs)
    return fail("the output position is not affine in the output index");
  // The span of the output layout. A dim of extent one has no pitch.
  int64_t span = 1;
  SmallVector<bool> pairedOutputDim(outputShape.size(), false);
  for (const Paired& p : paired) pairedOutputDim[p.outputDim] = true;
  for (auto [m, extent] : llvm::enumerate(outputShape)) {
    if (extent == 1) continue;
    if (pitch[m] <= 0) return fail("an output dim has no positive pitch");
    span += pitch[m] * (extent - 1);
  }

  // The elements the op reads that the input layout places.
  std::string loopBox;
  {
    std::string vars, bounds;
    for (auto [j, range] : llvm::enumerate(loopRanges)) {
      vars += llvm::formatv("{0}d{1}", j ? ", " : "", j).str();
      bounds += llvm::formatv("{0}0 <= d{1} <= {2}", j ? " and " : " : ", j,
                              range - 1)
                    .str();
    }
    loopBox = "{ [" + vars + "]" + bounds + " }";
  }
  isl_set* placed = isl_map_domain(toIslMap(inputLayout, ctx));
  isl_set* read = isl_set_intersect(
      isl_set_apply(isl_set_read_from_str(ctx, loopBox.c_str()),
                    toIslMap(*inputIndex, ctx)),
      isl_set_copy(placed));
  if (isl_set_is_empty(read) == isl_bool_true) {
    isl_set_free(read);
    isl_set_free(placed);
    return fail("the op reads no element the input layout places");
  }

  // The digits of an element: per paired result k, q_k = floor((i_k -
  // o_k) / s_k) and, for strides above one, the residue (i_k - o_k) mod s_k;
  // per class result, the index itself. o_k is the smallest index read.
  unsigned rank = inputMap.getNumResults();
  std::string vars;
  for (unsigned k = 0; k < rank; ++k) {
    vars += llvm::formatv("{0}i{1}", k ? ", " : "", k).str();
  }
  SmallVector<std::string> quotients, classDigits;
  for (const Paired& p : paired) {
    int64_t origin = getDimBound(read, p.result, /*max=*/false);
    std::string shifted = llvm::formatv("(i{0} - {1})", p.result, origin);
    quotients.push_back(
        llvm::formatv("floor({0} / {1})", shifted, p.stride).str());
    if (p.stride > 1) {
      classDigits.push_back(
          llvm::formatv("({0} mod {1})", shifted, p.stride).str());
    }
  }
  for (unsigned k : classes) {
    classDigits.push_back(llvm::formatv("i{0}", k).str());
  }
  // The range of each digit over the elements read.
  std::string digitList;
  for (const auto& digit : llvm::concat<std::string>(quotients, classDigits)) {
    digitList += (digitList.empty() ? "" : ", ") + digit;
  }
  std::string digitMapStr = "{ [" + vars + "] -> [" + digitList + "] }";
  isl_set* digits = isl_set_apply(
      isl_set_copy(read), isl_map_read_from_str(ctx, digitMapStr.c_str()));
  SmallVector<int64_t> lo, hi;
  for (unsigned d = 0; d < quotients.size() + classDigits.size(); ++d) {
    lo.push_back(getDimBound(digits, d, /*max=*/false));
    hi.push_back(getDimBound(digits, d, /*max=*/true));
  }
  isl_set_free(digits);
  for (auto [d, p] : llvm::enumerate(paired)) {
    if (hi[d] >= outputShape[p.outputDim]) {
      isl_set_free(read);
      isl_set_free(placed);
      return fail(
          "the elements read along an input dim span more strides than the "
          "output dim has");
    }
  }

  // kappa, the row-major index of the class digits, is placed where the
  // output layout puts the output dims the input does not read (e.g. a
  // convolution's output channel), in row-major order; the classes beyond
  // those take further copies of the output's span. Each copy is the image of
  // the output layout, so the placement is one-to-one, and
  //   f = sum_k pitch_k * q_k + sum_u pitch_u * u(kappa) + span * overflow.
  std::string kappa = "0", constraints;
  int64_t numClasses = 1;
  for (int64_t d = classDigits.size() - 1; d >= 0; --d) {
    unsigned pos = quotients.size() + d;
    kappa += llvm::formatv(" + {0} * ({1} - {2})", numClasses, classDigits[d],
                           lo[pos])
                 .str();
    constraints += llvm::formatv(" and {0} <= {1} <= {2}", lo[pos],
                                 classDigits[d], hi[pos])
                       .str();
    numClasses *= hi[pos] - lo[pos] + 1;
  }
  std::string flat;
  for (auto [d, p] : llvm::enumerate(paired)) {
    flat +=
        llvm::formatv(" + {0} * {1}", pitch[p.outputDim], quotients[d]).str();
    constraints += llvm::formatv(" and 0 <= {0} <= {1}", quotients[d],
                                 outputShape[p.outputDim] - 1)
                       .str();
  }
  int64_t unpairedSize = 1;
  for (int64_t m = outputShape.size() - 1; m >= 0; --m) {
    if (pairedOutputDim[m] || outputShape[m] == 1) continue;
    flat += llvm::formatv(" + {0} * (floor(({1}) / {2}) mod {3})", pitch[m],
                          kappa, unpairedSize, outputShape[m])
                .str();
    unpairedSize *= outputShape[m];
  }
  if (numClasses > unpairedSize) {
    flat +=
        llvm::formatv(" + {0} * floor(({1}) / {2})", span, kappa, unpairedSize)
            .str();
  }
  std::string layoutStr =
      llvm::formatv(
          "{ [{0}] -> [ct, slot] : {1}ct + slot = 0{2} and 0 <= "
          "slot <= {3}{4} }",
          vars, numSlots, flat, numSlots - 1, constraints)
          .str();
  isl_map* layout = isl_map_intersect_domain(
      isl_map_read_from_str(ctx, layoutStr.c_str()), placed);
  layout = simplify(layout);

  // The layout must place every element read, once each, in distinct slots,
  // and use no more ciphertexts than the input and output layouts do.
  isl_map* inputMapIsl = toIslMap(inputLayout, ctx);
  isl_map* outputMapIsl = toIslMap(outputLayout, ctx);
  int64_t budget = std::max(getNumCiphertexts(inputMapIsl),
                            getNumCiphertexts(outputMapIsl)) +
                   1;
  isl_map_free(inputMapIsl);
  isl_map_free(outputMapIsl);
  int64_t numCts = getNumCiphertexts(layout);
  isl_set* domain = isl_map_domain(isl_map_copy(layout));
  isl_bool covers = isl_set_is_subset(read, domain);
  isl_set_free(read);
  isl_set_free(domain);
  isl_bool injective = isl_map_is_injective(layout);
  isl_bool singleValued = isl_map_is_single_valued(layout);
  std::string why;
  if (covers != isl_bool_true)
    why = "the layout does not place every element read";
  else if (injective != isl_bool_true || singleValued != isl_bool_true)
    why = "the layout is not a one-to-one placement";
  else if (isl_map_n_basic_map(layout) != 1)
    why = "the layout is not a single basic map";
  else if (numCts > budget)
    why = llvm::formatv("the layout needs {0} ciphertexts, more than {1}",
                        numCts, budget)
              .str();
  if (!why.empty()) {
    isl_map_free(layout);
    return fail(why);
  }
  isl_basic_map* bmap = nullptr;
  isl_map_foreach_basic_map(
      layout,
      [](__isl_take isl_basic_map* b, void* user) {
        *static_cast<isl_basic_map**>(user) = b;
        return isl_stat_ok;
      },
      &bmap);
  isl_map_free(layout);
  // Frees bmap and ctx.
  return std::optional<IntegerRelation>(convertBasicMapToRelation(bmap));
}

}  // namespace heir
}  // namespace mlir
