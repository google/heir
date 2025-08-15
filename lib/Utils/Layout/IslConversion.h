#include "include/isl/ast_build.h"                                  // from @isl
#include "include/isl/ast_type.h"                                   // from @isl
#include "include/isl/constraint.h"                                 // from @isl
#include "include/isl/ctx.h"                                        // from @isl
#include "include/isl/local_space.h"                                // from @isl
#include "include/isl/map.h"                                        // from @isl
#include "include/isl/map_type.h"                                   // from @isl
#include "include/isl/set.h"                                        // from @isl
#include "include/isl/space.h"                                      // from @isl
#include "include/isl/space_type.h"                                 // from @isl
#include "include/isl/union_map.h"                                  // from @isl
#include "include/isl/union_map_type.h"                             // from @isl
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project

namespace mlir {
namespace heir {

__isl_give isl_basic_map* convertRelationToBasicMap(
    const presburger::IntegerRelation& rel, isl_ctx* ctx);

}  // namespace heir
}  // namespace mlir
