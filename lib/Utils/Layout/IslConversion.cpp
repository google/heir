#include "lib/Utils/Layout/IslConversion.h"

#include <cassert>
#include <cstdint>
#include <string>

#include "include/isl/aff.h"                     // from @isl
#include "include/isl/ctx.h"                     // from @isl
#include "include/isl/local_space.h"             // from @isl
#include "include/isl/map.h"                     // from @isl
#include "include/isl/map_type.h"                // from @isl
#include "include/isl/mat.h"                     // from @isl
#include "include/isl/set.h"                     // from @isl
#include "include/isl/space.h"                   // from @isl
#include "include/isl/space_type.h"              // from @isl
#include "include/isl/val.h"                     // from @isl
#include "include/isl/val_type.h"                // from @isl
#include "llvm/include/llvm/ADT/DynamicAPInt.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/Sequence.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"   // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project

#define DEBUG_TYPE "isl-conversion"

namespace mlir {
namespace heir {

using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

namespace {
// Borrowed from
// https://github.com/EnzymeAD/Enzyme-JAX/blob/ec61fe8bee1abb50ca3883cdded669e978624ad9/src/enzyme_ad/jax/Passes/SimplifyAffineExprs.cpp#L47
__isl_give isl_mat* setElement(__isl_take isl_mat* mat, isl_ctx* ctx, int row,
                               int col, int64_t val) {
  isl_val* v = isl_val_int_from_si(ctx, val);
  return isl_mat_set_element_val(mat, row, col, v);
}

__isl_give isl_mat* createConstraintRows(__isl_keep isl_ctx* ctx,
                                         const IntegerRelation& rel,
                                         bool isEq) {
  unsigned numRows = isEq ? rel.getNumEqualities() : rel.getNumInequalities();
  unsigned numDimVars = rel.getNumDimVars();
  unsigned numLocalVars = rel.getNumLocalVars();
  unsigned numSymbolVars = rel.getNumSymbolVars();

  unsigned numCols = rel.getNumCols();
  isl_mat* mat = isl_mat_alloc(ctx, numRows, numCols);

  for (unsigned row = 0; row < numRows; row++) {
    // Get coefficients for domain and range variables
    for (unsigned col = 0; col < numDimVars; col++) {
      mat = setElement(mat, ctx, row, col,
                       isEq ? rel.atEq64(row, col) : rel.atIneq64(row, col));
    }

    // Get coefficients for local variables
    for (unsigned col = 0; col < numLocalVars; col++) {
      unsigned fplCol = numDimVars + numSymbolVars + col;
      unsigned islCol = numDimVars + col;
      mat = setElement(
          mat, ctx, row, islCol,
          isEq ? rel.atEq64(row, fplCol) : rel.atIneq64(row, fplCol));
    }

    // Get coefficients for symbol variables
    for (unsigned col = 0; col < numSymbolVars; col++) {
      unsigned fplCol = numDimVars + col;
      unsigned islCol = numDimVars + numLocalVars + col;
      mat = setElement(
          mat, ctx, row, islCol,
          isEq ? rel.atEq64(row, fplCol) : rel.atIneq64(row, fplCol));
    }

    // Get constant term
    mat = setElement(
        mat, ctx, row, numCols - 1,
        isEq ? rel.atEq64(row, numCols - 1) : rel.atIneq64(row, numCols - 1));
  }
  return mat;
}

void populateConstraints(IntegerRelation& rel, __isl_keep isl_mat* mat,
                         bool eq) {
  unsigned numRows = isl_mat_rows(mat);
  unsigned numCols = isl_mat_cols(mat);
  for (unsigned row = 0; row < numRows; row++) {
    SmallVector<int64_t, 8> rowVals;
    rowVals.reserve(numCols);
    for (unsigned col = 0; col < numCols; col++) {
      isl_val* val = isl_mat_get_element_val(mat, row, col);
      int64_t valInt = isl_val_get_num_si(val);
      rowVals.push_back(valInt);
      isl_val_free(val);
    }
    if (eq) {
      rel.addEquality(rowVals);
    } else {
      rel.addInequality(rowVals);
    }
  }
}

}  // namespace

__isl_give isl_basic_map* convertRelationToBasicMap(const IntegerRelation& rel,
                                                    __isl_keep isl_ctx* ctx) {
  isl_mat* eqMat = createConstraintRows(ctx, rel, /*isEq=*/true);
  isl_mat* ineqMat = createConstraintRows(ctx, rel, /*isEq=*/false);
  LLVM_DEBUG({
    llvm::dbgs() << "Converting IntegerRelation to isl_basic_map\n";
    rel.dump();
    llvm::dbgs() << "\n";
    llvm::dbgs() << "eqMat rows:\n";
    isl_mat_dump(eqMat);
    llvm::dbgs() << "\n";
    llvm::dbgs() << "ineqMat rows:\n";
    isl_mat_dump(ineqMat);
    llvm::dbgs() << "\n";
  });

  unsigned numDomain = rel.getNumDomainVars();
  unsigned numDimVars = rel.getNumDimVars();
  unsigned numSymbols = rel.getNumSymbolVars();
  unsigned numLocals = rel.getNumLocalVars();
  unsigned localOffset = rel.getVarKindOffset(VarKind::Local);
  unsigned constCol = rel.getNumCols() - 1;

  // The constraints over (dims, locals) as a set, with the locals as ordinary
  // coordinates. Passing the locals to isl as divisions without definitions
  // would make isl eliminate them, which can be very slow on relations with
  // coupled divisions. Instead, each local that FPL can express as a floor
  // division of the other variables is substituted by that division, so isl
  // knows its definition, and only the remaining locals become existentially
  // quantified.
  isl_basic_set* constraints = isl_basic_set_from_constraint_matrices(
      isl_space_set_alloc(ctx, numSymbols, numDimVars + numLocals), eqMat,
      ineqMat, isl_dim_set, isl_dim_div, isl_dim_param, isl_dim_cst);

  presburger::DivisionRepr divs = rel.getLocalReprs();
  SmallVector<unsigned> unknown;
  for (unsigned k = 0; k < numLocals; ++k) {
    if (!divs.hasRepr(k)) unknown.push_back(k);
  }

  // The substitution (dims, unknown locals) -> (dims, locals).
  isl_local_space* ls = isl_local_space_from_space(
      isl_space_set_alloc(ctx, numSymbols, numDimVars + unknown.size()));
  SmallVector<isl_aff*> localAffs(numLocals, nullptr);
  for (auto [i, k] : llvm::enumerate(unknown)) {
    localAffs[k] = isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set,
                                         numDimVars + i);
  }
  // A division only refers to locals whose divisions FPL found before it, so
  // repeated passes define every local with a representation.
  bool progress = true;
  while (progress) {
    progress = false;
    for (unsigned k = 0; k < numLocals; ++k) {
      if (localAffs[k]) continue;
      ArrayRef<DynamicAPInt> dividend = divs.getDividend(k);
      bool ready = llvm::all_of(llvm::seq(0u, numLocals), [&](unsigned j) {
        return dividend[localOffset + j] == 0 || localAffs[j];
      });
      if (!ready) continue;
      isl_aff* numerator = isl_aff_val_on_domain(
          isl_local_space_copy(ls),
          isl_val_int_from_si(ctx, int64_t(dividend[constCol])));
      for (unsigned j = 0; j < numDimVars; ++j) {
        if (dividend[j] != 0)
          numerator = isl_aff_add_coefficient_val(
              numerator, isl_dim_in, j,
              isl_val_int_from_si(ctx, int64_t(dividend[j])));
      }
      for (unsigned j = 0; j < numSymbols; ++j) {
        if (dividend[numDimVars + j] != 0)
          numerator = isl_aff_add_coefficient_val(
              numerator, isl_dim_param, j,
              isl_val_int_from_si(ctx, int64_t(dividend[numDimVars + j])));
      }
      for (unsigned j = 0; j < numLocals; ++j) {
        if (dividend[localOffset + j] == 0) continue;
        numerator = isl_aff_add(
            numerator,
            isl_aff_scale_val(
                isl_aff_copy(localAffs[j]),
                isl_val_int_from_si(ctx, int64_t(dividend[localOffset + j]))));
      }
      localAffs[k] = isl_aff_floor(isl_aff_scale_down_val(
          numerator, isl_val_int_from_si(ctx, int64_t(divs.getDenom(k)))));
      progress = true;
    }
  }

  isl_multi_aff* substitution = isl_multi_aff_zero(isl_space_alloc(
      ctx, numSymbols, numDimVars + unknown.size(), numDimVars + numLocals));
  for (unsigned j = 0; j < numDimVars; ++j) {
    substitution = isl_multi_aff_set_aff(
        substitution, j,
        isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, j));
  }
  for (unsigned k = 0; k < numLocals; ++k) {
    assert(localAffs[k] && "division refers to a local without a definition");
    substitution =
        isl_multi_aff_set_aff(substitution, numDimVars + k, localAffs[k]);
  }
  isl_local_space_free(ls);

  constraints = isl_basic_set_preimage_multi_aff(constraints, substitution);
  constraints = isl_basic_set_project_out(constraints, isl_dim_set, numDimVars,
                                          unknown.size());
  isl_basic_map* result = isl_basic_map_from_range(constraints);
  return isl_basic_map_move_dims(result, isl_dim_in, 0, isl_dim_out, 0,
                                 numDomain);
}

presburger::IntegerRelation convertBasicMapToRelation(
    __isl_take isl_basic_map* bmap) {
  isl_ctx* ctx = isl_basic_map_get_ctx(bmap);
  // Variables in an IntegerRelation are stored in the order
  //
  //   Domain, Range, Symbols, Locals, Constant
  //
  // https://github.com/llvm/llvm-project/blob/8b091961b134661a3bbc95646a3a9b2344d684f8/mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h#L144-L145
  //
  // Because ISL provides this API that lets you choose the order of the
  // variables, we can copy these directly to FPL's IntMatrix.
  isl_mat* eqMat = isl_basic_map_equalities_matrix(
      bmap, isl_dim_in, isl_dim_out, isl_dim_param, isl_dim_div, isl_dim_cst);
  isl_mat* ineqMat = isl_basic_map_inequalities_matrix(
      bmap, isl_dim_in, isl_dim_out, isl_dim_param, isl_dim_div, isl_dim_cst);

  PresburgerSpace fplSpace = PresburgerSpace::getRelationSpace(
      /*numDomain=*/isl_basic_map_dim(bmap, isl_dim_in),
      /*numRange=*/isl_basic_map_dim(bmap, isl_dim_out),
      /*numSymbols=*/isl_basic_map_dim(bmap, isl_dim_param),
      /*numLocals=*/isl_basic_map_dim(bmap, isl_dim_div));
  IntegerRelation result(
      /*numReservedInequalities=*/isl_mat_rows(ineqMat),
      /*numReservedEqualities=*/isl_mat_rows(eqMat),
      /*numReservedCols=*/isl_mat_cols(eqMat), fplSpace);

  populateConstraints(result, eqMat, /*eq=*/true);
  populateConstraints(result, ineqMat, /*eq=*/false);

  isl_mat_free(eqMat);
  isl_mat_free(ineqMat);
  isl_basic_map_free(bmap);
  isl_ctx_free(ctx);

  return result;
}

FailureOr<presburger::IntegerRelation> getIntegerRelationFromIslStr(
    std::string islStr) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* bmap = isl_basic_map_read_from_str(ctx, islStr.c_str());
  if (!bmap) {
    isl_ctx_free(ctx);
    return failure();
  }
  return convertBasicMapToRelation(bmap);
}

}  // namespace heir
}  // namespace mlir
