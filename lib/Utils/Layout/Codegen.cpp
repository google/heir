#include "lib/Utils/Layout/Codegen.h"

#include <cstdint>

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
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

using presburger::IntegerRelation;
using presburger::VarKind;

static __isl_give isl_union_map* convertRelationToUnionMap(
    const IntegerRelation& rel, isl_ctx* ctx) {
  // ISL has these variables types, which map to MLIR FPL variable types:
  // isl_dim_param -> VarKind::Symbol
  // isl_dim_in -> VarKind::Domain
  // isl_dim_out -> VarKind::Range
  // isl_dim_div -> VarKind::Local
  //
  // ISL puts parameters first, but since we have zero symbols, we can ignore
  // the implied change to the offsets.
  unsigned numDomain = rel.getNumVarKind(VarKind::Domain);
  unsigned numRange = rel.getNumVarKind(VarKind::Range);
  unsigned numLocal = rel.getNumVarKind(VarKind::Local);

  // This represents the mapping from all four variables to just iterate over
  // the range variables. This is specific to FHE because we know the packing
  // must be a partial function from range to domain.
  //
  // E.g., this is specifying the `S[row,col,ct,slot] -> [ct,slot]` part of a
  // union map like
  //
  //   {
  //      S[row,col,ct,slot] -> [ct,slot] :
  //      0 <= row,ct < 4
  //      and 0 <= col < 8
  //      and 0 <= slot < 32
  //      and ((-row + slot) % 4) = 0 and (-col + ct + slot) % 8 = 0
  //   }
  //
  unsigned numIn = numDomain + numRange;
  unsigned numOut = numRange;
  isl_space* space = isl_space_alloc(ctx, /*n_params=*/0, numIn, numOut);

  // "S" is a hardcoded name for the inner-most statement that will be executed
  // to assign the data entry to the ciphertext slot. Users will be forced to
  // match on this name in the generated AST.
  //
  // E.g., S might be defined in plain C, using the layout example above, as
  //
  //   void S(int a, int b, int c, int d, int data[4][8], int slot[4][32]) {
  //     slot[c][d] = data[a][b];
  //   }
  //
  space = isl_space_set_tuple_name(space, isl_dim_in, "S");

  // Nb., consider someday using isl_space_set_dim_name to link these variables
  // to relevant constructs (like MLIR SSA variables, somehow?)

  isl_basic_map* bmap = isl_basic_map_universe(isl_space_copy(space));
  isl_local_space* ls = isl_local_space_from_space(space);
  if (numLocal > 0) {
    ls = isl_local_space_add_dims(ls, isl_dim_div, numLocal);
  }

  // Identical variables in the range and domain must be linked
  //
  // E.g., in S[row,col,ct,slot] -> [ct,slot], the ct and slot variables are
  // different and must be marked as the same.
  for (int i = 0; i < numRange; i++) {
    isl_constraint* c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
    c = isl_constraint_set_coefficient_si(c, isl_dim_in, numDomain + i, 1);
    c = isl_constraint_set_coefficient_si(c, isl_dim_out, i, -1);
    bmap = isl_basic_map_add_constraint(bmap, c);
  }

  // Now copy the coefficients of the constraints from the flattened IntegerSet
  // to ISL.
  auto copyConstraintsFromUnionMap = [&](bool isEquality) {
    unsigned numConstraints =
        isEquality ? rel.getNumEqualities() : rel.getNumInequalities();

    for (unsigned idx = 0; idx < numConstraints; ++idx) {
      SmallVector<int64_t> coeffs =
          isEquality ? rel.getEquality64(idx) : rel.getInequality64(idx);

      auto* spaceCopy = isl_local_space_copy(ls);
      isl_constraint* c = isEquality
                              ? isl_constraint_alloc_equality(spaceCopy)
                              : isl_constraint_alloc_inequality(spaceCopy);

      c = isl_constraint_set_constant_si(c, coeffs.back());

      for (int i = 0; i < numDomain; i++) {
        unsigned offset = i + rel.getVarKindOffset(VarKind::Domain);
        c = isl_constraint_set_coefficient_si(c, isl_dim_in, i, coeffs[offset]);
      }

      for (int i = 0; i < numRange; i++) {
        unsigned offset = i + rel.getVarKindOffset(VarKind::Range);
        c = isl_constraint_set_coefficient_si(c, isl_dim_in, numDomain + i,
                                              coeffs[offset]);
      }

      for (int i = 0; i < numLocal; i++) {
        unsigned offset = i + rel.getVarKindOffset(VarKind::Local);
        c = isl_constraint_set_coefficient_si(c, isl_dim_div, i,
                                              coeffs[offset]);
      }

      bmap = isl_basic_map_add_constraint(bmap, c);
    }
  };

  copyConstraintsFromUnionMap(/*isEquality=*/true);
  copyConstraintsFromUnionMap(/*isEquality=*/false);

  isl_local_space_free(ls);

  return isl_union_map_from_basic_map(bmap);
}

__isl_give isl_ast_node* constructAst(const presburger::IntegerRelation& rel,
                                      isl_ctx* ctx) {
  // The easiest way to convert an integer relation to an ISL schedule is
  // actually to write the ISL union-set as a string. This is because ISL's API
  // otherwise manually requires you to flatten constraints and remove
  // divs/mods.
  isl_union_map* schedule = convertRelationToUnionMap(rel, ctx);

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

}  // namespace heir
}  // namespace mlir
