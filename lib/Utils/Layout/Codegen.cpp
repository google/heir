#include "lib/Utils/Layout/Codegen.h"

#include "include/isl/ast_build.h"       // from @isl
#include "include/isl/ast_type.h"        // from @isl
#include "include/isl/constraint.h"      // from @isl
#include "include/isl/ctx.h"             // from @isl
#include "include/isl/local_space.h"     // from @isl
#include "include/isl/map.h"             // from @isl
#include "include/isl/map_type.h"        // from @isl
#include "include/isl/set.h"             // from @isl
#include "include/isl/space.h"           // from @isl
#include "include/isl/space_type.h"      // from @isl
#include "include/isl/union_map.h"       // from @isl
#include "include/isl/union_map_type.h"  // from @isl
#include "lib/Utils/Layout/IslConversion.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

using presburger::IntegerRelation;
using presburger::VarKind;

// Converts a basic map (representing proper domain/range vars as in/out vars)
// to a schedule that maps all domain+range vars to range vars.
//
// In particular, this tells the ISL codegen that we want to iterate over just
// the range variables. This is specific to FHE because we know the packing must
// be a partial function from range (ct, slot) to domain (data indices).
//
// E.g., for an input basic map like
//
//   {
//      [row,col] -> [ct,slot] :
//      0 <= row,ct < 4
//      and 0 <= col < 8
//      and 0 <= slot < 32
//      and ((-row + slot) % 4) = 0 and (-col + ct + slot) % 8 = 0
//   }
//
// The output schedule would have the same constraints, but the map would start
// with
//
//   S[row,col,ct,slot] -> [ct,slot]
//
static __isl_give isl_union_map* createSchedule(__isl_keep isl_basic_map* bmap,
                                                isl_ctx* ctx) {
  isl_basic_map* schedule_bmap = isl_basic_map_copy(bmap);
  unsigned numIn = isl_basic_map_dim(schedule_bmap, isl_dim_in);
  unsigned numOut = isl_basic_map_dim(schedule_bmap, isl_dim_out);

  // Insert two new dimensions for the original range variables into the domain.
  schedule_bmap =
      isl_basic_map_insert_dims(schedule_bmap, isl_dim_in, numIn, numOut);

  // Add constraints to equate the new domain dimensions with the original range
  // dimensions.
  for (unsigned i = 0; i < numOut; ++i) {
    isl_constraint* c = isl_constraint_alloc_equality(
        isl_basic_map_get_local_space(schedule_bmap));
    c = isl_constraint_set_coefficient_si(c, isl_dim_in, numIn + i, 1);
    c = isl_constraint_set_coefficient_si(c, isl_dim_out, i, -1);
    schedule_bmap = isl_basic_map_add_constraint(schedule_bmap, c);
  }

  // Set the domain tuple name to "S". This will be used in codegen for the
  // statement to be executed, e.g., S(a, b, c, d)
  schedule_bmap = isl_basic_map_set_tuple_name(schedule_bmap, isl_dim_in, "S");
  return isl_union_map_from_basic_map(schedule_bmap);
}

__isl_give isl_ast_node* constructAst(const presburger::IntegerRelation& rel,
                                      isl_ctx* ctx) {
  isl_basic_map* bmap = convertRelationToBasicMap(rel, ctx);
  isl_union_map* schedule = createSchedule(bmap, ctx);

  // Context and options are intentionally empty. We don't need any of these
  // features, though I admit I have not looked into what they can provide.
  isl_set* context = isl_set_universe(isl_space_params_alloc(ctx, 0));
  isl_union_map* options = isl_union_map_empty(isl_space_params_alloc(ctx, 0));

  // Build the AST
  isl_ast_build* build = isl_ast_build_from_context(context);
  build = isl_ast_build_set_options(build, options);
  isl_ast_node* tree = isl_ast_build_node_from_schedule_map(build, schedule);
  isl_ast_build_free(build);

  return tree;
}

FailureOr<isl_ast_node*> generateLoopNest(
    const presburger::IntegerRelation& rel, __isl_keep isl_ctx* ctx) {
  isl_ast_node* tree = constructAst(rel, ctx);
  if (!tree) {
    return failure();
  }
  return tree;
}

FailureOr<std::string> generateLoopNestAsCStr(
    const presburger::IntegerRelation& rel) {
  isl_ctx* ctx = isl_ctx_alloc();
  auto result = generateLoopNest(rel, ctx);
  if (failed(result)) {
    isl_ctx_free(ctx);
    return failure();
  }
  isl_ast_node* tree = result.value();
  char* cStr = isl_ast_node_to_C_str(tree);
  std::string actual = std::string(cStr);
  free(cStr);
  isl_ast_node_free(tree);
  isl_ctx_free(ctx);
  // Add a leading newline for ease of comparison with multiline strings.
  return actual.insert(0, "\n");
}

}  // namespace heir
}  // namespace mlir
