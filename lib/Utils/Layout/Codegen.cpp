#include "lib/Utils/Layout/Codegen.h"

#include "isl/ast_build.h"                                          // from @isl
#include "isl/ast_type.h"                                           // from @isl
#include "isl/ctx.h"                                                // from @isl
#include "isl/set.h"                                                // from @isl
#include "isl/space.h"                                              // from @isl
#include "isl/union_map.h"                                          // from @isl
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

static __isl_give isl_union_map* convertRelationToString(
    const presburger::IntegerRelation& rel, isl_ctx* ctx) {
  // FIXME: replace with actual string construction
  //  return isl_union_map_read_from_str(
  //      ctx,
  //      "{ S[row,col,ct,slot] -> [ct,slot] : "
  //      "0 <= row,ct < 4 and 0 <= col < 8 and 0 <= slot < 32 and "
  //      "((-row + slot) % 4) = 0 and (-col + ct + slot) % 8 = 0 }");
  return isl_union_map_read_from_str(
      ctx,
      "{ S[d0,d1] -> [d1] : "
      "0 <= d0 <= 10 and 0 <= d1 <= 10 and d0 - d1 = 0 }");
}

__isl_give isl_ast_node* constructAst(const presburger::IntegerRelation& rel,
                                      isl_ctx* ctx) {
  isl_union_map* schedule;
  isl_set* context;
  isl_union_map* options;
  isl_ast_build* build;
  isl_ast_node* tree;

  // The easiest way to convert an integer relation to an ISL schedule is
  // actually to write the ISL union-set as a string. This is because ISL's API
  // otherwise manually requires you to flatten constraints and remove
  // divs/mods.
  schedule = convertRelationToString(rel, ctx);

  // Context and options are intentionally empty. We don't need any of these
  // features, though I admit I have not looked into what they can provide.
  context = isl_set_universe(isl_space_params_alloc(ctx, 0));
  options = isl_union_map_empty(isl_space_params_alloc(ctx, 0));

  /* Build the AST */
  build = isl_ast_build_from_context(context);
  build = isl_ast_build_set_options(build, options);
  tree = isl_ast_build_node_from_schedule_map(build, schedule);
  isl_ast_build_free(build);

  return tree;
}

FailureOr<isl_ast_node*> generateLoopNest(
    const presburger::IntegerRelation& rel, isl_ctx* ctx) {
  isl_ast_node* tree = constructAst(rel, ctx);
  if (!tree) {
    return failure();
  }
  return tree;
}

}  // namespace heir
}  // namespace mlir
