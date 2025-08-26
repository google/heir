#include <cassert>
#include <cstdint>
#include <string>

#include "include/isl/ctx.h"                  // from @isl
#include "include/isl/map.h"                  // from @isl
#include "include/isl/map_type.h"             // from @isl
#include "include/isl/mat.h"                  // from @isl
#include "include/isl/space.h"                // from @isl
#include "include/isl/space_type.h"           // from @isl
#include "include/isl/val.h"                  // from @isl
#include "include/isl/val_type.h"             // from @isl
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

#define DEBUG_TYPE "isl-conversion"

namespace mlir {
namespace heir {

using presburger::IntegerRelation;
using presburger::PresburgerSpace;

namespace {
// Borrowed from
// https://github.com/EnzymeAD/Enzyme-JAX/blob/ec61fe8bee1abb50ca3883cdded669e978624ad9/src/enzyme_ad/jax/Passes/SimplifyAffineExprs.cpp#L47
__isl_give isl_mat* createConstraintRows(__isl_keep isl_ctx* ctx,
                                         const IntegerRelation& rel,
                                         bool isEq) {
  unsigned numRows = isEq ? rel.getNumEqualities() : rel.getNumInequalities();
  unsigned numDimVars = rel.getNumDimVars();
  unsigned numLocalVars = rel.getNumLocalVars();
  unsigned numSymbolVars = rel.getNumSymbolVars();

  unsigned numCols = rel.getNumCols();
  isl_mat* mat = isl_mat_alloc(ctx, numRows, numCols);

  for (unsigned i = 0; i < numRows; i++) {
    // Get the row based on isEq.
    auto row = isEq ? rel.getEquality(i) : rel.getInequality(i);

    assert(row.size() == numCols);

    // Dims stay at the same positions.
    for (unsigned j = 0; j < numDimVars; j++)
      mat = isl_mat_set_element_si(mat, i, j, (int64_t)row[j]);
    // Output local ids before symbols.
    for (unsigned j = 0; j < numLocalVars; j++)
      mat = isl_mat_set_element_si(
          mat, i, j + numDimVars, (int64_t)row[j + numDimVars + numSymbolVars]);
    // Output symbols in the end.
    for (unsigned j = 0; j < numSymbolVars; j++)
      mat = isl_mat_set_element_si(mat, i, j + numDimVars + numLocalVars,
                                   (int64_t)row[j + numDimVars]);
    // Finally outputs the constant.
    mat =
        isl_mat_set_element_si(mat, i, numCols - 1, (int64_t)row[numCols - 1]);
  }
  return mat;
}

void populateConstraints(IntegerRelation& rel, __isl_keep isl_mat* mat,
                         bool eq) {
  unsigned numRows = isl_mat_rows(mat);
  unsigned numCols = isl_mat_cols(mat);

  for (unsigned i = 0; i < numRows; i++) {
    SmallVector<int64_t, 8> row;
    for (unsigned j = 0; j < numCols; j++) {
      isl_val* val = isl_mat_get_element_val(mat, i, j);
      row.push_back(isl_val_get_num_si(val));
      isl_val_free(val);
    }

    if (eq) {
      rel.addEquality(row);
    } else {
      rel.addInequality(row);
    }
  }
}

}  // namespace

__isl_give isl_basic_map* convertRelationToBasicMap(const IntegerRelation& rel,
                                                    __isl_keep isl_ctx* ctx) {
  isl_mat* eqMat = createConstraintRows(ctx, rel, /*isEq=*/true);
  isl_mat* ineqMat = createConstraintRows(ctx, rel, /*isEq=*/false);
  LLVM_DEBUG({
    llvm::dbgs() << "Adding domain relation\n";
    llvm::dbgs() << " ISL eq mat:\n";
    isl_mat_dump(eqMat);
    llvm::dbgs() << " ISL ineq mat:\n";
    isl_mat_dump(ineqMat);
    llvm::dbgs() << "\n";
  });

  isl_space* space =
      isl_space_alloc(ctx, rel.getNumSymbolVars(), rel.getNumDomainVars(),
                      rel.getNumRangeVars());
  return isl_basic_map_from_constraint_matrices(
      space, eqMat, ineqMat, isl_dim_in, isl_dim_out, isl_dim_div,
      isl_dim_param, isl_dim_cst);
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
