#include <string>

#include "include/isl/ctx.h"                                        // from @isl
#include "include/isl/map_type.h"                                   // from @isl
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Converts a presburger::IntegerRelation to an isl_basic_map. Requires the
// user to manage the input isl_ctx.
__isl_give isl_basic_map* convertRelationToBasicMap(
    const presburger::IntegerRelation& rel, __isl_keep isl_ctx* ctx);

// Converts a an isl_basic_map to presburger::IntegerRelation. This function
// frees the input isl_basic_map and its underlying isl_ctx before returning
// the IntegerRelation.
presburger::IntegerRelation convertBasicMapToRelation(
    __isl_take isl_basic_map* bmap);

FailureOr<presburger::IntegerRelation> getIntegerRelationFromIslStr(
    std::string islStr);

}  // namespace heir
}  // namespace mlir
