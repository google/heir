#include "lib/Utils/Layout/GenericKernel.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
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
#include "include/isl/aff.h"          // from @isl
#include "include/isl/ctx.h"          // from @isl
#include "include/isl/local_space.h"  // from @isl
#include "include/isl/map.h"          // from @isl
#include "include/isl/map_type.h"     // from @isl
#include "include/isl/mat.h"          // from @isl
#include "include/isl/point.h"        // from @isl
#include "include/isl/set.h"          // from @isl
#include "include/isl/space.h"        // from @isl
#include "include/isl/space_type.h"   // from @isl
#include "include/isl/val.h"          // from @isl

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

// Converts `map`, from the elements of a tensor to (ct, slot), to a layout
// relation. The map must place every element in `required`, no two in one slot,
// as a single basic map; `domain` is the domain of `map`. Consumes `map`,
// `domain`, `required` (any may be null), and `ctx`.
FailureOr<IntegerRelation> toLayoutRelation(
    isl_ctx* ctx, __isl_take isl_map* map, __isl_take isl_set* domain,
    __isl_take isl_set* required, StringRef name, std::string& reason) {
  auto fail = [&](const Twine& why) -> FailureOr<IntegerRelation> {
    reason = why.str();
    isl_map_free(map);
    isl_set_free(domain);
    isl_set_free(required);
    isl_ctx_free(ctx);
    return failure();
  };
  auto islError = [&]() { return fail("ISL failed"); };
  if (!map || !domain || !required) return islError();
  isl_bool total = isl_set_is_subset(required, domain);
  isl_set_free(required);
  required = nullptr;
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

// An affine map whose linear results are evaluated without walking the
// AffineExpr tree, for use inside the groupInstances loop nest.
struct CompiledAffineExpr {
  int64_t constant = 0;
  SmallVector<std::pair<unsigned, int64_t>, 4> terms;
  AffineExpr fallback;

  int64_t eval(ArrayRef<int64_t> dims) const {
    if (fallback) return evaluate(fallback, dims);
    int64_t res = constant;
    for (auto [d, c] : terms) res += c * dims[d];
    return res;
  }
};

struct CompiledAffineMap {
  SmallVector<CompiledAffineExpr, 4> exprs;

  static CompiledAffineMap get(AffineMap map, unsigned numDims) {
    CompiledAffineMap result;
    result.exprs.reserve(map.getNumResults());
    for (AffineExpr expr : map.getResults()) {
      CompiledAffineExpr compiled;
      SmallVector<int64_t, 8> coeffs(numDims, 0);
      int64_t cst = 0;
      auto extractLinear = [&](auto& self, AffineExpr e,
                               int64_t scale) -> bool {
        if (auto dim = dyn_cast<AffineDimExpr>(e)) {
          if (dim.getPosition() >= numDims) return false;
          coeffs[dim.getPosition()] += scale;
          return true;
        }
        if (auto c = dyn_cast<AffineConstantExpr>(e)) {
          cst += scale * c.getValue();
          return true;
        }
        auto bin = dyn_cast<AffineBinaryOpExpr>(e);
        if (!bin) return false;
        if (bin.getKind() == AffineExprKind::Add) {
          return self(self, bin.getLHS(), scale) &&
                 self(self, bin.getRHS(), scale);
        }
        if (bin.getKind() == AffineExprKind::Mul) {
          if (auto rc = dyn_cast<AffineConstantExpr>(bin.getRHS())) {
            return self(self, bin.getLHS(), scale * rc.getValue());
          }
          if (auto lc = dyn_cast<AffineConstantExpr>(bin.getLHS())) {
            return self(self, bin.getRHS(), scale * lc.getValue());
          }
        }
        return false;
      };
      if (extractLinear(extractLinear, expr, 1)) {
        compiled.constant = cst;
        for (unsigned j = 0; j < numDims; ++j) {
          if (coeffs[j] != 0) compiled.terms.emplace_back(j, coeffs[j]);
        }
      } else {
        compiled.fallback = expr;
      }
      result.exprs.push_back(std::move(compiled));
    }
    return result;
  }

  int64_t evalLinearized(ArrayRef<int64_t> strides,
                         ArrayRef<int64_t> dims) const {
    int64_t flat = 0;
    for (size_t k = 0; k < exprs.size(); ++k) {
      flat += exprs[k].eval(dims) * strides[k];
    }
    return flat;
  }
};

// The (ct, slot) positions a layout places each data element at.
class SlotTable {
 public:
  static SlotTable get(const presburger::IntegerRelation& layout);

  // The positions of the element at `map(dims)`, sorted by (slot, ct); none
  // for elements the layout does not place.
  ArrayRef<std::pair<int64_t, int64_t>> lookup(const CompiledAffineMap& map,
                                               ArrayRef<int64_t> dims) const {
    int64_t flat = 0;
    for (size_t k = 0; k < shape.size(); ++k) {
      int64_t idx = map.exprs[k].eval(dims);
      int64_t dim = shape[k];
      if (idx < 0 || idx >= dim) return {};
      flat = flat * dim + idx;
    }
    return slots[flat];
  }

  // The positions of every element.
  ArrayRef<SmallVector<std::pair<int64_t, int64_t>, 1>> positions() const {
    return slots;
  }

  // Whether every position has ct >= 0 and slot in [0, numSlots).
  bool fitsIn(int64_t numSlots) const;

 private:
  int64_t flatten(ArrayRef<int64_t> index) const;

  SmallVector<int64_t> shape;
  std::vector<SmallVector<std::pair<int64_t, int64_t>, 1>> slots;
};

struct PmaPieces {
  struct Piece {
    isl_set* dom;
    isl_multi_aff* maff;
  };
  std::vector<Piece> pieces;

  explicit PmaPieces(__isl_take isl_pw_multi_aff* pma) {
    isl_pw_multi_aff_foreach_piece(
        pma,
        [](__isl_take isl_set* dom, __isl_take isl_multi_aff* maff,
           void* u) -> isl_stat {
          static_cast<std::vector<Piece>*>(u)->push_back({dom, maff});
          return isl_stat_ok;
        },
        &pieces);
    isl_pw_multi_aff_free(pma);
  }
  ~PmaPieces() {
    for (Piece& p : pieces) {
      isl_set_free(p.dom);
      isl_multi_aff_free(p.maff);
    }
  }
  PmaPieces(const PmaPieces&) = delete;
  PmaPieces& operator=(const PmaPieces&) = delete;
};

// Concatenates the range of `a` and `b` on each pair of pieces without calling
// isl_multi_aff_gist.
__isl_give isl_pw_multi_aff* flatRangeProductNoGist(
    __isl_take isl_pw_multi_aff* a, __isl_take isl_pw_multi_aff* b) {
  isl_space* space = isl_space_range_product(isl_pw_multi_aff_get_space(a),
                                             isl_pw_multi_aff_get_space(b));
  space = isl_space_flatten_range(space);
  isl_pw_multi_aff* res = isl_pw_multi_aff_empty(space);
  PmaPieces piecesA(a);
  PmaPieces piecesB(b);
  for (const auto& pa : piecesA.pieces) {
    for (const auto& pb : piecesB.pieces) {
      isl_set* dom =
          isl_set_intersect(isl_set_copy(pa.dom), isl_set_copy(pb.dom));
      isl_multi_aff* maff = isl_multi_aff_flat_range_product(
          isl_multi_aff_copy(pa.maff), isl_multi_aff_copy(pb.maff));
      res = isl_pw_multi_aff_union_add(res, isl_pw_multi_aff_alloc(dom, maff));
    }
  }
  return res;
}

// Pulls back `pma` by `inner` on each pair of pieces without calling
// isl_multi_aff_gist.
__isl_give isl_pw_multi_aff* pullbackNoGist(
    __isl_take isl_pw_multi_aff* pma, __isl_take isl_pw_multi_aff* inner) {
  isl_space* space = isl_space_map_from_domain_and_range(
      isl_space_domain(isl_pw_multi_aff_get_space(inner)),
      isl_space_range(isl_pw_multi_aff_get_space(pma)));
  isl_pw_multi_aff* res = isl_pw_multi_aff_empty(space);
  PmaPieces outerPieces(pma);
  PmaPieces innerPieces(inner);
  for (const auto& po : outerPieces.pieces) {
    for (const auto& pi : innerPieces.pieces) {
      isl_set* pre = isl_set_preimage_multi_aff(isl_set_copy(po.dom),
                                                isl_multi_aff_copy(pi.maff));
      isl_set* dom = isl_set_intersect(isl_set_copy(pi.dom), pre);
      isl_multi_aff* maff = isl_multi_aff_pullback_multi_aff(
          isl_multi_aff_copy(po.maff), isl_multi_aff_copy(pi.maff));
      res = isl_pw_multi_aff_union_add(res, isl_pw_multi_aff_alloc(dom, maff));
    }
  }
  return res;
}

}  // namespace

SlotTable SlotTable::get(const IntegerRelation& layout) {
  unsigned rank = layout.getNumDomainVars();
  PointPairCollector collector(rank, 2);
  enumeratePoints(layout, collector);
  SlotTable table;
  table.shape.assign(rank, 0);
  for (const auto& [index, position] : collector.points) {
    for (unsigned i = 0; i < rank; ++i) {
      assert(index[i] >= 0 && "expected non-negative domain coordinate");
      table.shape[i] = std::max(table.shape[i], index[i] + 1);
    }
  }
  int64_t size = 1;
  for (int64_t dim : table.shape) size *= dim;
  table.slots.resize(size);
  for (const auto& [index, position] : collector.points) {
    table.slots[table.flatten(index)].push_back({position[0], position[1]});
  }
  for (auto& positions : table.slots) {
    llvm::sort(positions, [](const auto& a, const auto& b) {
      return std::tie(a.second, a.first) < std::tie(b.second, b.first);
    });
    positions.erase(llvm::unique(positions), positions.end());
  }
  return table;
}

bool SlotTable::fitsIn(int64_t numSlots) const {
  for (const auto& positions : slots) {
    for (auto [ct, slot] : positions) {
      if (ct < 0 || slot < 0 || slot >= numSlots) return false;
    }
  }
  return true;
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
  size_t numSecretInputs = 0;
  for (auto [i, input] : llvm::enumerate(spec.inputs)) {
    if (input.isCleartext()) {
      strides[i] = computeStrides(*input.cleartextShape);
      continue;
    }
    ++numSecretInputs;
    inputSlots[i] = SlotTable::get(input.layout);
    if (!inputSlots[i]->fitsIn(numSlots)) {
      reason = llvm::formatv("input {0} has a position outside the slots", i);
      return failure();
    }
  }
  if (!outputSlots.fitsIn(numSlots)) {
    reason = "the output layout has a position outside the slots";
    return failure();
  }

  // Walk the loop nest. Each (iteration, output slot) instance has a key
  // [ct_o, (ct_i, shift_i) per secret input], taking the replica with the
  // lexicographically smallest (shift_i, ct_i) of a replicated input element
  // and ct_i = -1 for one the layout does not place (read as zero). Per key
  // and output slot, collect the distinct cleartext reads (row-major indices)
  // with their multiplicities.
  InstanceGroups groups;
  auto& readsByKey = groups.readsByKey;
  auto& coverage = groups.coverage;
  const unsigned numLoops = spec.loopRanges.size();
  CompiledAffineMap outputMap =
      CompiledAffineMap::get(spec.output.indexingMap, numLoops);
  std::vector<CompiledAffineMap> inputMaps(spec.inputs.size());
  for (auto [i, input] : llvm::enumerate(spec.inputs)) {
    inputMaps[i] = CompiledAffineMap::get(input.indexingMap, numLoops);
  }
  SmallVector<int64_t> d(numLoops, 0);
  SmallVector<int64_t> cleartextReads;
  std::vector<ArrayRef<std::pair<int64_t, int64_t>>> inPositions(
      spec.inputs.size());
  SmallVector<size_t> inCursors(spec.inputs.size(), 0);
  std::vector<int64_t> key(1 + 2 * numSecretInputs, 0);
  // Sparse keys stay in readsByKey directly; keys that touch at least
  // kDenseSlotPromotionThreshold slots promote to a flat numSlots-sized vector
  // for O(1) slot indexing during the loop nest and flush back into readsByKey
  // afterwards.
  constexpr size_t kDenseSlotPromotionThreshold = 64;
  struct KeySlots {
    std::map<int64_t, InstanceGroups::Reads>* sparse = nullptr;
    std::vector<InstanceGroups::Reads> dense;

    InstanceGroups::Reads& get(int64_t slot, int64_t numSlots) {
      if (!dense.empty()) return dense[slot];
      auto& res = (*sparse)[slot];
      if (sparse->size() >= kDenseSlotPromotionThreshold) {
        dense.resize(numSlots);
        for (auto& [s, r] : *sparse) dense[s] = std::move(r);
        sparse->clear();
        return dense[slot];
      }
      return res;
    }
  };
  std::map<std::vector<int64_t>, KeySlots> keySlots;
  std::vector<int64_t> lastKey;
  KeySlots* currentSlots = nullptr;

  for (bool done = llvm::is_contained(spec.loopRanges, 0); !done;) {
    ArrayRef<std::pair<int64_t, int64_t>> outPositions =
        outputSlots.lookup(outputMap, d);
    if (!outPositions.empty()) {
      cleartextReads.clear();
      for (auto [i, input] : llvm::enumerate(spec.inputs)) {
        if (input.isCleartext()) {
          cleartextReads.push_back(inputMaps[i].evalLinearized(strides[i], d));
        } else {
          inPositions[i] = inputSlots[i]->lookup(inputMaps[i], d);
          inCursors[i] = 0;
        }
      }
      for (auto [outputCt, outputSlot] : outPositions) {
        key[0] = outputCt;
        size_t keyPos = 1;
        for (auto [i, input] : llvm::enumerate(spec.inputs)) {
          if (input.isCleartext()) continue;
          ArrayRef<std::pair<int64_t, int64_t>> pos = inPositions[i];
          if (pos.empty()) {
            key[keyPos++] = -1;
            key[keyPos++] = 0;
            continue;
          }
          // outPositions is sorted by slot, so the cursor only moves forward.
          size_t& cur = inCursors[i];
          while (cur < pos.size() && pos[cur].second < outputSlot) ++cur;
          if (cur < pos.size()) {
            key[keyPos++] = pos[cur].first;
            key[keyPos++] = pos[cur].second - outputSlot;
          } else {
            key[keyPos++] = pos[0].first;
            key[keyPos++] = pos[0].second + numSlots - outputSlot;
          }
        }
        if (!currentSlots || key != lastKey) {
          lastKey = key;
          currentSlots = &keySlots[key];
          if (!currentSlots->sparse) currentSlots->sparse = &readsByKey[key];
        }
        InstanceGroups::Reads& reads = currentSlots->get(outputSlot, numSlots);
        auto it = llvm::find_if(reads, [&](const auto& read) {
          return read.first == cleartextReads;
        });
        if (it == reads.end()) {
          reads.push_back({cleartextReads, 1});
        } else {
          ++it->second;
        }
      }
    }
    // Advance the iteration in row-major order.
    int64_t j = static_cast<int64_t>(numLoops) - 1;
    while (j >= 0 && ++d[j] == spec.loopRanges[j]) d[j--] = 0;
    done = j < 0;
  }
  for (auto& [k, ks] : keySlots) {
    if (ks.dense.empty()) continue;
    for (int64_t s = 0; s < numSlots; ++s) {
      if (!ks.dense[s].empty()) {
        ks.sparse->emplace_hint(ks.sparse->end(), s, std::move(ks.dense[s]));
      }
    }
  }

  coverage.assign(numCts, std::vector<int64_t>(numSlots, 0));
  for (const auto& [k, slots] : readsByKey) {
    if (k[0] >= numCts) {
      reason =
          "the output layout writes more ciphertexts than the result holds";
      return failure();
    }
    for (const auto& [slot, reads] : slots) {
      for (const auto& [read, count] : reads) {
        coverage[k[0]][slot] += count;
        if (kind == GenericReductionKind::Mul && count > 1) {
          reason =
              "a multiplicative reduction cannot repeat a factor in a slot";
          return failure();
        }
      }
      if (kind == GenericReductionKind::None && coverage[k[0]][slot] > 1) {
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

FailureOr<GroupCount> countInstanceGroups(const GenericKernelSpec& spec,
                                          std::string& reason) {
  if (failed(validateGenericKernelSpec(spec, reason))) return failure();
  IslContext isl;
  isl_ctx* ctx = isl.ctx;
  auto fail = [&](StringRef why) -> FailureOr<GroupCount> {
    reason = why.str();
    return failure();
  };
  const unsigned numLoops = spec.loopRanges.size();

  // The instances (d, ct_o, slot_o) place the output element of iteration d.
  // Taking the preimage of the layout under the indexing map, rather than
  // composing them, adds no existential variables.
  FailureOr<IntegerRelation> outputIndex =
      getIndexingMapRelation(spec.output.indexingMap, spec.loopRanges);
  if (failed(outputIndex))
    return fail("indexing map of the output is not affine");
  isl_set* rawInstances =
      isl_set_flatten(isl_map_wrap(isl_map_preimage_domain_pw_multi_aff(
          toIslMap(spec.output.layout, ctx),
          isl_pw_multi_aff_from_map(toIslMap(*outputIndex, ctx)))));
  // Lift local floor-division variables of the output layout into set
  // dimensions before (ct_o, slot_o) so Gaussian elimination pivots on slot_o
  // (unit coefficient) rather than on the output layout's modulus quotient.
  unsigned numOutDivs = 0;
  isl_set* instances = rawInstances;
  if (isl_set_n_basic_set(rawInstances) == 1) {
    struct LiftInstances {
      unsigned numLoops;
      unsigned numOutDivs = 0;
      isl_set* res = nullptr;
    } liftInst{numLoops, 0, nullptr};
    isl_set_foreach_basic_set(
        rawInstances,
        [](__isl_take isl_basic_set* bset, void* u) -> isl_stat {
          auto* d = static_cast<LiftInstances*>(u);
          bset = isl_basic_set_flatten(isl_basic_set_lift(bset));
          unsigned total = isl_basic_set_dim(bset, isl_dim_set);
          d->numOutDivs = total - (d->numLoops + 2);
          isl_set* s = isl_set_from_basic_set(bset);
          if (d->numOutDivs > 0) {
            s = isl_set_move_dims(s, isl_dim_param, 0, isl_dim_set, d->numLoops,
                                  2);
            s = isl_set_move_dims(s, isl_dim_set, d->numLoops + d->numOutDivs,
                                  isl_dim_param, 0, 2);
          }
          d->res = s;
          return isl_stat_ok;
        },
        &liftInst);
    isl_set_free(rawInstances);
    instances = liftInst.res;
    numOutDivs = liftInst.numOutDivs;
  }
  const unsigned instDim = numLoops + numOutDivs;
  isl_space* instanceSpace = isl_set_get_space(instances);
  auto instanceCoord = [&](unsigned pos) {
    return isl_pw_aff_var_on_domain(
        isl_local_space_from_space(isl_space_copy(instanceSpace)), isl_dim_set,
        pos);
  };
  // The key [ct_o, (ct_i, shift_i) per secret input] of an instance.
  isl_pw_multi_aff* key = isl_pw_multi_aff_from_pw_aff(instanceCoord(instDim));
  unsigned numSecretInputs = 0;
  for (auto [i, input] : llvm::enumerate(spec.inputs)) {
    if (input.isCleartext()) continue;
    FailureOr<IntegerRelation> index =
        getIndexingMapRelation(input.indexingMap, spec.loopRanges);
    if (failed(index)) {
      isl_pw_multi_aff_free(key);
      isl_set_free(instances);
      isl_space_free(instanceSpace);
      return fail(
          llvm::formatv("indexing map of input {0} is not affine", i).str());
    }
    isl_map* layout = toIslMap(input.layout, ctx);

    // The instance (d, div_out..., ct_o, slot_o) reads element index(d) into
    // slot_o.
    isl_pw_multi_aff* element =
        pullbackNoGist(isl_pw_multi_aff_from_map(toIslMap(*index, ctx)),
                       isl_pw_multi_aff_project_out_map(
                           isl_space_copy(instanceSpace), isl_dim_set, numLoops,
                           numOutDivs + 2));

    // Express the input layout in instance space with output coordinates
    // [shift, c], where shift = s - slot_o - shiftOffset. Working in the lifted
    // instance space lets Gaussian elimination substitute slot_o directly into
    // the input layout's slot congruence.
    isl_map* instLayout = isl_map_preimage_domain_pw_multi_aff(layout, element);
    auto toShiftedMap = [&](__isl_take isl_map* m, __isl_take isl_set* dom,
                            int64_t shiftOffset,
                            bool fixZeroShift) -> isl_map* {
      isl_set* flat = isl_set_flatten(isl_map_wrap(m));
      isl_space* flatSpace = isl_set_get_space(flat);
      isl_local_space* ls =
          isl_local_space_from_space(isl_space_copy(flatSpace));
      isl_multi_aff* psi =
          isl_multi_aff_identity(isl_space_map_from_set(flatSpace));
      isl_aff* soAff = isl_aff_var_on_domain(isl_local_space_copy(ls),
                                             isl_dim_set, instDim + 1);
      isl_aff* shiftAff = isl_aff_var_on_domain(isl_local_space_copy(ls),
                                                isl_dim_set, instDim + 2);
      isl_aff* cAff = isl_aff_var_on_domain(ls, isl_dim_set, instDim + 3);
      isl_aff* sExpr = isl_aff_add(shiftAff, soAff);
      if (shiftOffset != 0) sExpr = isl_aff_add_constant_si(sExpr, shiftOffset);
      psi = isl_multi_aff_set_aff(psi, instDim + 2, cAff);
      psi = isl_multi_aff_set_aff(psi, instDim + 3, sExpr);
      flat = isl_set_preimage_multi_aff(flat, psi);
      isl_map* shifted = isl_map_from_domain(flat);
      shifted = isl_map_move_dims(shifted, isl_dim_out, 0, isl_dim_in,
                                  instDim + 2, 2);
      if (fixZeroShift) {
        shifted = isl_map_fix_si(shifted, isl_dim_out, 0, 0);
      } else if (shiftOffset == 0) {
        shifted = isl_map_lower_bound_si(shifted, isl_dim_out, 0, 0);
      }
      return isl_map_intersect_domain(shifted, dom);
    };

    auto swapShiftAndCt =
        [&](__isl_take isl_pw_multi_aff* pma,
            __isl_take isl_set* restrictDom) -> isl_pw_multi_aff* {
      struct SwapData {
        isl_set* restrictDom;
        isl_pw_multi_aff* res;
      } sdata{restrictDom,
              isl_pw_multi_aff_empty(isl_pw_multi_aff_get_space(pma))};
      isl_pw_multi_aff_foreach_piece(
          pma,
          [](__isl_take isl_set* dom, __isl_take isl_multi_aff* maff,
             void* u) -> isl_stat {
            auto* d = static_cast<SwapData*>(u);
            isl_aff* shift = isl_multi_aff_get_aff(maff, 0);
            isl_aff* c = isl_multi_aff_get_aff(maff, 1);
            maff = isl_multi_aff_set_aff(maff, 0, c);
            maff = isl_multi_aff_set_aff(maff, 1, shift);
            if (d->restrictDom) {
              dom = isl_set_intersect(dom, isl_set_copy(d->restrictDom));
            }
            maff = isl_multi_aff_gist(maff, isl_set_copy(dom));
            d->res = isl_pw_multi_aff_union_add(
                d->res, isl_pw_multi_aff_alloc(dom, maff));
            return isl_stat_ok;
          },
          &sdata);
      isl_set_free(restrictDom);
      isl_pw_multi_aff_free(pma);
      return sdata.res;
    };

    // Solve lexmin on the convex `instances` set rather than on subtracted
    // subdomains, which are non-convex unions of basic sets. First check if
    // shift = 0 covers all placed instances (e.g. a pulled-back layout, where
    // the equality s = slot_o eliminates slot_o before PIP); otherwise solve
    // shift >= 0 and wrapped shift >= -numSlots on `instances`.
    isl_pw_multi_aff* inputKey =
        isl_pw_multi_aff_empty(isl_space_alloc(ctx, 0, instDim + 2, 2));
    isl_map* sameSlot = toShiftedMap(isl_map_copy(instLayout),
                                     isl_set_copy(instances), 0, true);
    isl_pw_multi_aff* zeroKey = isl_map_lexmin_pw_multi_aff(sameSlot);
    isl_set* rem = isl_set_subtract(
        isl_set_copy(instances),
        isl_pw_multi_aff_domain(isl_pw_multi_aff_copy(zeroKey)));
    bool onlyZeroShift = isl_set_is_empty(rem) == isl_bool_true;
    if (!onlyZeroShift) {
      isl_map* anyOnRem = toShiftedMap(
          isl_map_copy(instLayout), isl_set_copy(rem), -spec.numSlots, false);
      onlyZeroShift = isl_map_is_empty(anyOnRem) == isl_bool_true;
      isl_map_free(anyOnRem);
    }
    if (onlyZeroShift) {
      inputKey = isl_pw_multi_aff_union_add(inputKey,
                                            swapShiftAndCt(zeroKey, nullptr));
    } else {
      isl_pw_multi_aff_free(zeroKey);
      isl_set_free(rem);
      rem = isl_set_copy(instances);
      for (int64_t offset : {int64_t{0}, -spec.numSlots}) {
        if (isl_set_is_empty(rem) == isl_bool_true) break;
        isl_map* cand = toShiftedMap(isl_map_copy(instLayout),
                                     isl_set_copy(instances), offset, false);
        isl_pw_multi_aff* piece = isl_map_lexmin_pw_multi_aff(cand);
        isl_set* nextRem = isl_set_subtract(
            isl_set_copy(rem),
            isl_pw_multi_aff_domain(isl_pw_multi_aff_copy(piece)));
        isl_set* pieceDom = (offset == 0) ? nullptr : rem;
        if (offset == 0) isl_set_free(rem);
        inputKey = isl_pw_multi_aff_union_add(inputKey,
                                              swapShiftAndCt(piece, pieceDom));
        rem = nextRem;
      }
    }
    isl_map_free(instLayout);
    if (isl_set_is_empty(rem) != isl_bool_true) {
      isl_multi_val* zeroVal =
          isl_multi_val_zero(isl_space_set_alloc(ctx, 0, 2));
      zeroVal = isl_multi_val_set_val(zeroVal, 0, isl_val_negone(ctx));
      inputKey = isl_pw_multi_aff_union_add(
          inputKey, isl_pw_multi_aff_multi_val_on_domain(rem, zeroVal));
    } else {
      isl_set_free(rem);
    }
    key = flatRangeProductNoGist(key, inputKey);
    ++numSecretInputs;
  }
  isl_space_free(instanceSpace);
  if (numSecretInputs == 0) {
    key = isl_pw_multi_aff_intersect_domain(key, instances);
  } else {
    isl_set_free(instances);
  }

  struct Collector {
    unsigned numCoords;
    std::vector<SmallVector<int64_t, 4>> keys;
  } collector{1 + 2 * numSecretInputs, {}};
  struct PieceCtx {
    Collector* collector;
    isl_multi_aff* maff;
  };
  isl_stat status = isl_pw_multi_aff_foreach_piece(
      key,
      [](__isl_take isl_set* dom, __isl_take isl_multi_aff* maff,
         void* user) -> isl_stat {
        auto* collector = static_cast<Collector*>(user);
        PieceCtx pctx{collector, maff};
        isl_stat s = isl_set_foreach_basic_set(
            dom,
            [](__isl_take isl_basic_set* bset, void* u) -> isl_stat {
              auto* pctx = static_cast<PieceCtx*>(u);
              auto* collector = pctx->collector;
              if (isl_basic_set_is_empty(bset) == isl_bool_true) {
                isl_basic_set_free(bset);
                return isl_stat_ok;
              }
              isl_ctx* ctx = isl_basic_set_get_ctx(bset);
              unsigned inDim = isl_basic_set_dim(bset, isl_dim_set);
              unsigned nDiv = isl_basic_set_dim(bset, isl_dim_div);
              isl_multi_aff* bMaff = isl_multi_aff_copy(pctx->maff);

              // Substitute div-free equalities with a unit-coefficient pivot
              // into `bMaff`.
              assert(isl_basic_set_dim(bset, isl_dim_param) == 0 &&
                     "expected parameter-free instance basic set");
              isl_mat* eqMat = isl_basic_set_equalities_matrix(
                  bset, isl_dim_set, isl_dim_div, isl_dim_param, isl_dim_cst);
              unsigned nEq = isl_mat_rows(eqMat);
              unsigned cstCol = inDim + nDiv;
              SmallVector<bool, 12> usedPivot(inDim, false);
              for (int r = static_cast<int>(nEq) - 1; r >= 0; --r) {
                bool hasDiv = false;
                for (unsigned k = 0; k < nDiv; ++k) {
                  isl_val* v = isl_mat_get_element_val(eqMat, r, inDim + k);
                  bool nz = !isl_val_is_zero(v);
                  isl_val_free(v);
                  if (nz) {
                    hasDiv = true;
                    break;
                  }
                }
                if (hasDiv) continue;
                int pivotCol = -1;
                int64_t pivotSign = 0;
                for (int j = static_cast<int>(inDim) - 1; j >= 0; --j) {
                  isl_val* v = isl_mat_get_element_val(eqMat, r, j);
                  int64_t c = isl_val_get_num_si(v);
                  isl_val_free(v);
                  if (c != 0 && usedPivot[j]) {
                    pivotCol = -1;
                    break;
                  }
                  if (pivotCol < 0 && (c == 1 || c == -1)) {
                    pivotCol = j;
                    pivotSign = c;
                  }
                }
                if (pivotCol < 0) continue;
                usedPivot[pivotCol] = true;
                isl_local_space* ls = isl_local_space_from_space(
                    isl_space_set_alloc(ctx, 0, inDim));
                isl_val* cstV = isl_mat_get_element_val(eqMat, r, cstCol);
                int64_t negSign = -pivotSign;
                isl_aff* rhs = isl_aff_val_on_domain(
                    isl_local_space_copy(ls),
                    isl_val_int_from_si(ctx,
                                        negSign * isl_val_get_num_si(cstV)));
                isl_val_free(cstV);
                for (unsigned j = 0; j < inDim; ++j) {
                  if (static_cast<int>(j) == pivotCol) continue;
                  isl_val* v = isl_mat_get_element_val(eqMat, r, j);
                  int64_t c = isl_val_get_num_si(v);
                  isl_val_free(v);
                  if (c != 0) {
                    rhs = isl_aff_add_coefficient_si(rhs, isl_dim_in, j,
                                                     negSign * c);
                  }
                }
                isl_local_space_free(ls);
                isl_multi_aff* rowSubst = isl_multi_aff_identity(
                    isl_space_alloc(ctx, 0, inDim, inDim));
                rowSubst = isl_multi_aff_set_aff(rowSubst, pivotCol, rhs);
                bMaff = isl_multi_aff_pullback_multi_aff(bMaff, rowSubst);
              }
              isl_mat_free(eqMat);
              bMaff = isl_multi_aff_gist(
                  bMaff, isl_set_from_basic_set(isl_basic_set_copy(bset)));

              if (isl_multi_aff_involves_dims(bMaff, isl_dim_in, 0, inDim) ==
                  isl_bool_false) {
                SmallVector<int64_t, 4> coords(collector->numCoords);
                bool allConstInt = true;
                for (unsigned i = 0; i < collector->numCoords; ++i) {
                  isl_aff* aff = isl_multi_aff_get_aff(bMaff, i);
                  isl_val* val = isl_aff_get_constant_val(aff);
                  if (isl_aff_dim(aff, isl_dim_div) == 0 &&
                      isl_val_is_int(val) == isl_bool_true) {
                    coords[i] = isl_val_get_num_si(val);
                  } else {
                    allConstInt = false;
                  }
                  isl_val_free(val);
                  isl_aff_free(aff);
                }
                if (allConstInt) {
                  collector->keys.push_back(std::move(coords));
                  isl_basic_set_free(bset);
                  isl_multi_aff_free(bMaff);
                  return isl_stat_ok;
                }
              }

              // Project out dimensions of `bset` that `bMaff` does not depend
              // on when doing so does not introduce local divs.
              for (int j = static_cast<int>(inDim) - 1; j >= 0; --j) {
                if (isl_multi_aff_involves_dims(bMaff, isl_dim_in, j, 1) ==
                    isl_bool_true) {
                  continue;
                }
                unsigned prevDiv = isl_basic_set_dim(bset, isl_dim_div);
                isl_basic_set* proj = isl_basic_set_project_out(
                    isl_basic_set_copy(bset), isl_dim_set, j, 1);
                if (isl_basic_set_dim(proj, isl_dim_div) <= prevDiv) {
                  isl_basic_set_free(bset);
                  bset = proj;
                  bMaff = isl_multi_aff_drop_dims(bMaff, isl_dim_in, j, 1);
                  --inDim;
                } else {
                  isl_basic_set_free(proj);
                }
              }

              struct LinearFloor {
                int64_t constant = 0;
                int64_t denom = 1;
                SmallVector<std::pair<unsigned, int64_t>, 4> terms;
                int64_t eval(ArrayRef<int64_t> vals) const {
                  __int128 num = constant;
                  for (auto [idx, coeff] : terms)
                    num += __int128(coeff) * vals[idx];
                  return denom == 1 ? int64_t(num)
                                    : int64_t(num / denom - (num % denom < 0));
                }
              };
              unsigned remDim = isl_basic_set_dim(bset, isl_dim_set);
              auto parseOne = [remDim](__isl_keep isl_aff* a,
                                       std::optional<unsigned> maxDivIdx =
                                           std::nullopt) {
                LinearFloor lf;
                isl_val* dVal = isl_aff_get_denominator_val(a);
                lf.denom = isl_val_get_num_si(dVal);
                isl_aff* scaled = isl_aff_scale_val(isl_aff_copy(a), dVal);
                isl_val* cVal = isl_aff_get_constant_val(scaled);
                lf.constant = isl_val_get_num_si(cVal);
                isl_val_free(cVal);
                for (auto [type, offset] :
                     {std::pair{isl_dim_in, 0u}, {isl_dim_div, remDim}}) {
                  unsigned n = isl_aff_dim(scaled, type);
                  for (unsigned j = 0; j < n; ++j) {
                    isl_val* v = isl_aff_get_coefficient_val(scaled, type, j);
                    int64_t c = isl_val_get_num_si(v);
                    isl_val_free(v);
                    if (c != 0) {
                      assert((type != isl_dim_div || !maxDivIdx ||
                              j < *maxDivIdx) &&
                             "expected topological div ordering");
                      lf.terms.push_back({offset + j, c});
                    }
                  }
                }
                isl_aff_free(scaled);
                return lf;
              };
              SmallVector<unsigned, 8> activeDims;
              for (unsigned j = 0; j < remDim; ++j) {
                if (isl_multi_aff_involves_dims(bMaff, isl_dim_in, j, 1) ==
                    isl_bool_true) {
                  activeDims.push_back(j);
                }
              }
              unsigned maxDivs = 0;
              SmallVector<SmallVector<LinearFloor, 4>, 4> compiled;
              compiled.reserve(collector->numCoords);
              for (unsigned i = 0; i < collector->numCoords; ++i) {
                isl_aff* aff = isl_multi_aff_get_aff(bMaff, i);
                unsigned aDiv = isl_aff_dim(aff, isl_dim_div);
                maxDivs = std::max(maxDivs, aDiv);
                SmallVector<LinearFloor, 4> chain;
                chain.reserve(aDiv + 1);
                // ISL stores local divs in `aff`'s shared local space in
                // topological order, so `isl_aff_get_div(aff, k)` only
                // references divs `< k`, which have already been evaluated
                // into `sctx->pt[remDim + j]`.
                for (unsigned k = 0; k < aDiv; ++k) {
                  isl_aff* d = isl_aff_get_div(aff, k);
                  chain.push_back(parseOne(d, k));
                  isl_aff_free(d);
                }
                chain.push_back(parseOne(aff));
                compiled.push_back(std::move(chain));
                isl_aff_free(aff);
              }
              isl_multi_aff_free(bMaff);
              if (isl_basic_set_dim(bset, isl_dim_div) > 0) {
                bset = isl_basic_set_flatten(isl_basic_set_lift(bset));
              }

              struct ScanCtx {
                Collector* collector;
                unsigned remDim;
                const SmallVector<unsigned, 8>* activeDims;
                const SmallVector<SmallVector<LinearFloor, 4>, 4>* compiled;
                SmallVector<int64_t, 16> pt;
              } sctx{collector, remDim, &activeDims, &compiled,
                     SmallVector<int64_t, 16>(remDim + maxDivs, 0)};
              isl_set* scanSet = isl_set_from_basic_set(bset);
              isl_stat scanStat = isl_set_foreach_point(
                  scanSet,
                  [](__isl_take isl_point* point, void* su) -> isl_stat {
                    auto* sctx = static_cast<ScanCtx*>(su);
                    auto* collector = sctx->collector;
                    for (unsigned j : *sctx->activeDims) {
                      isl_val* v =
                          isl_point_get_coordinate_val(point, isl_dim_set, j);
                      sctx->pt[j] = isl_val_get_num_si(v);
                      isl_val_free(v);
                    }
                    isl_point_free(point);
                    SmallVector<int64_t, 4> coords(collector->numCoords);
                    for (unsigned i = 0; i < collector->numCoords; ++i) {
                      const auto& chain = (*sctx->compiled)[i];
                      for (unsigned k = 0; k + 1 < chain.size(); ++k)
                        sctx->pt[sctx->remDim + k] = chain[k].eval(sctx->pt);
                      coords[i] = chain.back().eval(sctx->pt);
                    }
                    collector->keys.push_back(std::move(coords));
                    return isl_stat_ok;
                  },
                  &sctx);
              isl_set_free(scanSet);
              return scanStat;
            },
            &pctx);
        isl_set_free(dom);
        isl_multi_aff_free(maff);
        return s;
      },
      &collector);
  isl_pw_multi_aff_free(key);
  if (status != isl_stat_ok) return fail("ISL failed to enumerate the keys");

  llvm::sort(collector.keys);
  collector.keys.erase(
      std::unique(collector.keys.begin(), collector.keys.end()),
      collector.keys.end());
  std::set<std::tuple<unsigned, int64_t, int64_t>> rotations;
  for (const auto& k : collector.keys) {
    for (unsigned i = 0; i < numSecretInputs; ++i) {
      if (k[2 + 2 * i] != 0) rotations.insert({i, k[1 + 2 * i], k[2 + 2 * i]});
    }
  }
  return GroupCount{static_cast<int64_t>(collector.keys.size()),
                    static_cast<int64_t>(rotations.size())};
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

  if (failed(validateLayout(StructuredOperand{inputMap, inputLayout}, "input",
                            reason))) {
    return fail(reason);
  }
  FailureOr<IntegerRelation> inputIndex =
      getIndexingMapRelation(inputMap, loopRanges);
  FailureOr<IntegerRelation> outputIndex =
      getIndexingMapRelation(outputMap, loopRanges);
  if (failed(inputIndex) || failed(outputIndex)) {
    return fail("the indexing maps are not affine");
  }
  isl_set* domain =
      getElementsWithPlacedPartner(*outputIndex, *inputIndex, inputLayout, ctx);
  isl_set* box = getBoxSet(outputShape, ctx);
  isl_bool total = isl_set_is_subset(box, domain);
  isl_set_free(box);
  if (total != isl_bool_true) {
    isl_set_free(domain);
    return fail(total == isl_bool_error
                    ? "ISL failed"
                    : "the induced relation does not place every element");
  }
  // Two output elements that read the same placed input element (without
  // reductions) or the same reduction fiber (with reductions) will share its
  // slot regardless of the input layout. Checking this in data-index space
  // avoids running parametric lexmin over the input layout's local variables.
  isl_map* outToIn = isl_map_apply_domain(toIslMap(*inputIndex, ctx),
                                          toIslMap(*outputIndex, ctx));
  isl_map* checkMap =
      reduceToLexmin ? isl_map_lexmin(outToIn)
                     : isl_map_intersect_range(
                           outToIn, isl_map_domain(toIslMap(inputLayout, ctx)));
  isl_bool canBeInjective = isl_map_is_injective(checkMap);
  isl_map_free(checkMap);
  if (canBeInjective != isl_bool_true) {
    isl_set_free(domain);
    return fail(canBeInjective == isl_bool_error
                    ? "ISL failed"
                    : "the induced relation places two elements in one slot");
  }

  FailureOr<isl_map*> inputAccess =
      getAccessMap(StructuredOperand{inputMap, inputLayout}, loopRanges, ctx,
                   "input", reason);
  if (failed(inputAccess)) {
    isl_set_free(domain);
    return fail(reason);
  }
  // o -> (ct, slot): every output element lands where the input element read
  // in the same iteration lives.
  isl_map* induced =
      isl_map_apply_domain(*inputAccess, toIslMap(*outputIndex, ctx));
  if (reduceToLexmin) induced = isl_map_lexmin(induced);
  return toLayoutRelation(ctx, isl_map_coalesce(induced), domain,
                          getBoxSet(outputShape, ctx), "induced", reason);
}

FailureOr<IntegerRelation> pullBackInputLayout(
    ArrayRef<int64_t> loopRanges, ArrayRef<bool> isReduction,
    AffineMap inputMap, ArrayRef<int64_t> inputShape, AffineMap outputMap,
    const IntegerRelation& outputLayout, std::string& reason) {
  // convertBasicMapToRelation frees this context on success, so every other
  // ISL object must be released before that call.
  isl_ctx* ctx = isl_ctx_alloc();
  auto fail = [&](std::string why) -> FailureOr<IntegerRelation> {
    reason = std::move(why);
    isl_ctx_free(ctx);
    return failure();
  };

  FailureOr<IntegerRelation> inputIndex =
      getIndexingMapRelation(inputMap, loopRanges);
  FailureOr<IntegerRelation> outputIndex =
      getIndexingMapRelation(outputMap, loopRanges);
  FailureOr<isl_map*> output =
      getAccessMap(StructuredOperand{outputMap, outputLayout}, loopRanges, ctx,
                   "the output", reason);
  if (failed(output)) return fail(reason);
  if (failed(inputIndex) || failed(outputIndex)) {
    isl_map_free(*output);
    return fail("the indexing maps are not affine");
  }
  isl_map* pulled = nullptr;
  if (llvm::any_of(loopRanges, [](int64_t r) { return r <= 0; })) {
    isl_map_free(*output);
    pulled = isl_map_empty(isl_space_alloc(ctx, 0, inputShape.size(), 2));
  } else {
    // Check whether the output layout uses only ciphertext 0 (co == 0).
    // When co == 0, c_in = r in [0, R - 1], so the outermost reduction loop
    // d_{r_0} is directly floor(c_in / (R / extent_0)) without introducing a
    // redundant top quotient floor(c_in / R).
    isl_map* coPos =
        isl_map_lower_bound_si(isl_map_copy(*output), isl_dim_out, 0, 1);
    isl_map* coNeg =
        isl_map_upper_bound_si(isl_map_copy(*output), isl_dim_out, 0, -1);
    bool singleOutputCt = isl_map_is_empty(coPos) == isl_bool_true &&
                          isl_map_is_empty(coNeg) == isl_bool_true;
    isl_map_free(coPos);
    isl_map_free(coNeg);
    if (singleOutputCt) {
      *output = isl_map_fix_si(*output, isl_dim_out, 0, 0);
    }

    // First project out the parallel loops while the reduction loops and
    // (co, so) are still domain dimensions (to the left of isl_dim_div), so
    // Gaussian elimination pivots on the parallel loops rather than on the
    // reduction loops. Then substitute the floor-division bijection
    // phi: (c_in, so) -> (d_red..., co, so) where c_in = co * R + r.
    unsigned numLoops = loopRanges.size();
    isl_set* outSet = isl_set_flatten(isl_map_wrap(*output));
    isl_map* combined = isl_map_intersect_domain(
        isl_map_add_dims(toIslMap(*inputIndex, ctx), isl_dim_in, 2), outSet);
    unsigned numRed = 0;
    for (int64_t j = static_cast<int64_t>(numLoops) - 1; j >= 0; --j) {
      if (isReduction[j]) {
        ++numRed;
      } else {
        combined = isl_map_project_out(combined, isl_dim_in, j, 1);
      }
    }

    isl_local_space* ls =
        isl_local_space_from_space(isl_space_set_alloc(ctx, 0, 2));
    isl_multi_aff* phi =
        isl_multi_aff_zero(isl_space_alloc(ctx, 0, 2, numRed + 2));
    isl_aff* cIn =
        isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, 0);
    isl_aff* q = isl_aff_copy(cIn);
    int64_t numReductions = 1;
    unsigned redPos = numRed;
    for (int64_t j = static_cast<int64_t>(numLoops) - 1; j >= 0; --j) {
      if (!isReduction[j]) continue;
      int64_t extent = loopRanges[j];
      numReductions *= extent;
      if (redPos == 1 && singleOutputCt) {
        phi = isl_multi_aff_set_aff(phi, --redPos, q);
        q = isl_aff_zero_on_domain(isl_local_space_copy(ls));
      } else {
        isl_aff* nextQ = isl_aff_floor(isl_aff_scale_down_val(
            isl_aff_copy(cIn), isl_val_int_from_si(ctx, numReductions)));
        isl_aff* dj =
            isl_aff_sub(q, isl_aff_scale_val(isl_aff_copy(nextQ),
                                             isl_val_int_from_si(ctx, extent)));
        phi = isl_multi_aff_set_aff(phi, --redPos, dj);
        q = nextQ;
      }
    }
    isl_aff_free(cIn);
    phi = isl_multi_aff_set_aff(phi, numRed, q);
    phi = isl_multi_aff_set_aff(phi, numRed + 1,
                                isl_aff_var_on_domain(ls, isl_dim_set, 1));

    combined = isl_map_preimage_domain_multi_aff(combined, phi);
    combined = isl_map_lower_bound_si(combined, isl_dim_in, 0, 0);
    if (singleOutputCt) {
      combined =
          isl_map_upper_bound_si(combined, isl_dim_in, 0, numReductions - 1);
    }
    pulled = isl_map_coalesce(isl_map_reverse(combined));
  }
  // Only elements the kernel actually reads need a placement; e.g., a stride-2
  // 1x1 conv never touches odd positions and has no equations for them.
  isl_set* required = llvm::any_of(loopRanges, [](int64_t r) { return r <= 0; })
                          ? getBoxSet(inputShape, ctx)
                          : isl_map_range(toIslMap(*inputIndex, ctx));
  isl_set* domain = getElementsWithPlacedPartner(*inputIndex, *outputIndex,
                                                 outputLayout, ctx);
  return toLayoutRelation(ctx, pulled, domain, required, "pulled-back", reason);
}

}  // namespace heir
}  // namespace mlir
