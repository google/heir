// Times each tier of isRelationEqual separately, on the layout pairs that
// motivated its current shape.
//

#include <cassert>
#include <functional>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "benchmark/isl/relations.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"

// ISL
#include "include/isl/ctx.h"                                        // from @isl
#include "include/isl/map.h"                                        // from @isl
#include "include/isl/map_type.h"                                   // from @isl
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using presburger::IntegerRelation;

// ---------------------------------------------------------------------------
// Tiers
// ---------------------------------------------------------------------------

// Each tier records what it decided in a "result" counter, so a run doubles as
// a report of which tier can actually settle which pair.
using Tier =
    std::function<double(const IntegerRelation&, const IntegerRelation&)>;

double tierObviouslyEqual(const IntegerRelation& lhs,
                          const IntegerRelation& rhs) {
  return lhs.isObviouslyEqual(rhs);
}

double tierProveUnequalByVolume(const IntegerRelation& lhs,
                                const IntegerRelation& rhs) {
  return succeeded(tryProveUnequalByVolume(lhs, rhs));
}

// isl's equality test, on both relations converted into a shared context. This
// mirrors what isRelationEqual does internally; it is a separate tier so its
// cost can be read apart from the checks that precede it.
double tierIslIsEqual(const IntegerRelation& lhs, const IntegerRelation& rhs) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl_map* map1 = isl_map_from_basic_map(convertRelationToBasicMap(lhs, ctx));
  isl_map* map2 = isl_map_from_basic_map(convertRelationToBasicMap(rhs, ctx));
  isl_bool equal = isl_map_is_equal(map1, map2);
  isl_map_free(map1);
  isl_map_free(map2);
  isl_ctx_free(ctx);
  return equal == isl_bool_true;
}

double tierIsRelationEqual(const IntegerRelation& lhs,
                           const IntegerRelation& rhs) {
  return isRelationEqual(lhs, rhs);
}

void runTier(benchmark::State& state, const Tier& tier,
             const IntegerRelation& lhs, const IntegerRelation& rhs) {
  double result = 0;
  for (auto _ : state) {
    result = tier(lhs, rhs);
    benchmark::DoNotOptimize(result);
  }
  state.counters["result"] = result;
}

// ---------------------------------------------------------------------------
// Pairs
// ---------------------------------------------------------------------------

// Rewrites a relation through isl, preserving its point set while giving isl a
// chance to pick a different constraint representation. Writing an equal
// relation by hand does not work: isl canonicalizes it straight back, and
// isObviouslyEqual then settles the pair before any tier runs.
IntegerRelation islRoundtrip(const IntegerRelation& rel) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* bmap = convertRelationToBasicMap(rel, ctx);
  bmap =
      isl_basic_map_remove_redundancies(isl_basic_map_detect_equalities(bmap));
  // convertBasicMapToRelation frees both the basic map and ctx.
  return convertBasicMapToRelation(bmap);
}

IntegerRelation parseOrDie(const char* islStr) {
  FailureOr<IntegerRelation> relation = getIntegerRelationFromIslStr(islStr);
  assert(succeeded(relation) && "benchmark relation failed to parse");
  return relation.value();
}

struct NamedPair {
  std::string name;
  IntegerRelation lhs;
  IntegerRelation rhs;
};

std::vector<NamedPair> buildPairs() {
  std::vector<NamedPair> pairs;

  // Synthetic: a large local-variable-heavy conv filter layout against a
  // point-set preserving rewrite of itself. No compile was observed comparing
  // these, but it bounds how badly the tiers behave on a relation much larger
  // than the one below.
  IntegerRelation largeFilter = parseOrDie(kLayout34Relation);
  pairs.push_back({"LargeFilterEqual", largeFilter, islRoundtrip(largeFilter)});

  // Two different large conv filter layouts (48 out / 32 in, kW 9 vs kW 1).
  pairs.push_back(
      {"LargeFilterUnequal", largeFilter, parseOrDie(kLayout29Relation)});

  // The layout that motivated this work, against a point-set preserving rewrite
  // of itself. The counterpart it was actually compared against in the compile
  // is not known, so this stands in for the equal-but-differently-written case.
  IntegerRelation nestedFloor = parseOrDie(kNestedFloorRelation);
  pairs.push_back({"NestedFloorEqual", nestedFloor, islRoundtrip(nestedFloor)});

  pairs.push_back({"NestedFloorUnequal", nestedFloor,
                   parseOrDie(kNestedFloorUnequalRelation)});

  return pairs;
}

void registerBenchmarks() {
  // Leaked deliberately: the relations must outlive RunSpecifiedBenchmarks.
  auto* pairs = new std::vector<NamedPair>(buildPairs());

  const std::pair<const char*, Tier> tiers[] = {
      {"ObviouslyEqual", tierObviouslyEqual},
      {"ProveUnequalByVolume", tierProveUnequalByVolume},
      {"IslIsEqual", tierIslIsEqual},
      {"IsRelationEqual", tierIsRelationEqual},
  };

  for (const NamedPair& pair : *pairs) {
    for (const auto& [tierName, tier] : tiers) {
      benchmark::RegisterBenchmark(
          "BM_" + std::string(tierName) + "/" + pair.name,
          [&pair, tier](benchmark::State& state) {
            runTier(state, tier, pair.lhs, pair.rhs);
          })
          ->Unit(benchmark::kMillisecond);
    }
  }
}

}  // namespace
}  // namespace heir
}  // namespace mlir

int main(int argc, char** argv) {
  mlir::heir::registerBenchmarks();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
