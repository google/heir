#include "include/isl/ctx.h"                                        // from @isl
#include "include/isl/map_type.h"                                   // from @isl
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project

namespace mlir {
namespace heir {

__isl_give isl_basic_map* convertRelationToBasicMap(
    const presburger::IntegerRelation& rel, isl_ctx* ctx);

}  // namespace heir
}  // namespace mlir
