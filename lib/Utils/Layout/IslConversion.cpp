#include "lib/Utils/Layout/IslConversion.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

#include "include/isl/aff.h"                    // from @isl
#include "include/isl/ctx.h"                    // from @isl
#include "include/isl/local_space.h"            // from @isl
#include "include/isl/map.h"                    // from @isl
#include "include/isl/map_type.h"               // from @isl
#include "include/isl/mat.h"                    // from @isl
#include "include/isl/set.h"                    // from @isl
#include "include/isl/space.h"                  // from @isl
#include "include/isl/space_type.h"             // from @isl
#include "include/isl/val.h"                    // from @isl
#include "include/isl/val_type.h"               // from @isl
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"    // from @llvm-project
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

  for (unsigned i = 0; i < numRows; i++) {
    // Get the row based on isEq.
    auto row = isEq ? rel.getEquality(i) : rel.getInequality(i);

    assert(row.size() == numCols);

    // Dims stay at the same positions.
    for (unsigned j = 0; j < numDimVars; j++)
      mat = setElement(mat, ctx, i, j, (int64_t)row[j]);
    // Output local ids before symbols.
    for (unsigned j = 0; j < numLocalVars; j++)
      mat = setElement(mat, ctx, i, j + numDimVars,
                       (int64_t)row[j + numDimVars + numSymbolVars]);
    // Output symbols in the end.
    for (unsigned j = 0; j < numSymbolVars; j++)
      mat = setElement(mat, ctx, i, j + numDimVars + numLocalVars,
                       (int64_t)row[j + numDimVars]);
    // Finally outputs the constant.
    mat = setElement(mat, ctx, i, numCols - 1, (int64_t)row[numCols - 1]);
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

  unsigned numDomain = rel.getNumDomainVars();
  unsigned numSymbols = rel.getNumSymbolVars();
  unsigned numLocals = rel.getNumLocalVars();
  unsigned numDimVars = rel.getNumDimVars();
  unsigned localOffset = rel.getVarKindOffset(presburger::VarKind::Local);
  unsigned constCol = rel.getNumCols() - 1;

  // Detect local variables defined by floor-division inequality pairs
  // 0 <= expr + c - d * q <= w (with 0 <= w <= d - 1) in topological order so
  // they can be bound as known ISL floor-divs before Gaussian/Bezout div
  // elimination runs on the equalities.
  struct FloorDivInfo {
    unsigned ineqRow;
    int64_t divisor;
  };
  SmallVector<std::optional<FloorDivInfo>> floorDivs(numLocals);
  SmallVector<unsigned> order;
  SmallVector<bool> scheduled(numLocals, false);
  bool progress = true;
  while (progress) {
    progress = false;
    for (unsigned k = 0; k < numLocals; ++k) {
      if (scheduled[k]) continue;
      unsigned col = localOffset + k;
      for (unsigned r1 = 0; r1 < rel.getNumInequalities() && !floorDivs[k];
           ++r1) {
        int64_t negD = rel.atIneq64(r1, col);
        if (negD >= -1 || negD == std::numeric_limits<int64_t>::min()) continue;
        bool ready = true;
        for (unsigned j = 0; j < numLocals; ++j) {
          if (j != k && rel.atIneq64(r1, localOffset + j) != 0 &&
              !scheduled[j]) {
            ready = false;
            break;
          }
        }
        if (!ready) continue;
        int64_t d = -negD;
        for (unsigned r2 = 0; r2 < rel.getNumInequalities(); ++r2) {
          if (rel.atIneq64(r2, col) != d) continue;
          __int128 w = static_cast<__int128>(rel.atIneq64(r1, constCol)) +
                       static_cast<__int128>(rel.atIneq64(r2, constCol));
          if (w < 0 || w > d - 1) continue;
          bool match = true;
          for (unsigned j = 0; j < constCol; ++j) {
            if (j != col &&
                static_cast<__int128>(rel.atIneq64(r1, j)) +
                        static_cast<__int128>(rel.atIneq64(r2, j)) !=
                    0) {
              match = false;
              break;
            }
          }
          if (match) {
            floorDivs[k] = FloorDivInfo{r1, d};
            break;
          }
        }
      }
      if (floorDivs[k].has_value()) {
        scheduled[k] = true;
        order.push_back(k);
        progress = true;
      }
    }
  }

  // When every local variable is a floor-division and at least two floor-divs
  // are coupled (either in a defining inequality or in an equality row),
  // pulling back the lifted basic set by the explicit floor-div `isl_multi_aff`
  // avoids ISL's Hermite/Bezout div elimination on the coupled equalities.
  // When every floor-div and equality involves at most one local variable, or
  // when non-floor-div existential local variables are present,
  // `isl_basic_map_from_constraint_matrices` eliminates them directly.
  bool hasCoupledLocals = false;
  if (order.size() == numLocals) {
    for (unsigned k : order) {
      unsigned r1 = floorDivs[k]->ineqRow;
      for (unsigned j = 0; j < numLocals; ++j) {
        if (j != k && rel.atIneq64(r1, localOffset + j) != 0) {
          hasCoupledLocals = true;
          break;
        }
      }
    }
    for (unsigned r = 0; r < rel.getNumEqualities() && !hasCoupledLocals; ++r) {
      unsigned floorDivsInRow = 0;
      for (unsigned k : order) {
        if (rel.atEq64(r, localOffset + k) != 0) ++floorDivsInRow;
      }
      if (floorDivsInRow >= 2) hasCoupledLocals = true;
    }
  }

  if (!order.empty() && hasCoupledLocals) {
    unsigned fullSetDim = numDimVars + numLocals;
    isl_local_space* ls = isl_local_space_from_space(
        isl_space_set_alloc(ctx, numSymbols, numDimVars));
    SmallVector<isl_aff*> localAffs(numLocals, nullptr);
    for (unsigned k : order) {
      unsigned r1 = floorDivs[k]->ineqRow;
      int64_t d = floorDivs[k]->divisor;
      isl_aff* num = isl_aff_val_on_domain(
          isl_local_space_copy(ls),
          isl_val_int_from_si(ctx, rel.atIneq64(r1, constCol)));
      for (unsigned j = 0; j < numDimVars; ++j) {
        int64_t c = rel.atIneq64(r1, j);
        if (c != 0) num = isl_aff_add_coefficient_si(num, isl_dim_in, j, c);
      }
      for (unsigned s = 0; s < numSymbols; ++s) {
        int64_t c = rel.atIneq64(r1, numDimVars + s);
        if (c != 0) num = isl_aff_add_coefficient_si(num, isl_dim_param, s, c);
      }
      for (unsigned j = 0; j < numLocals; ++j) {
        if (j == k) continue;
        int64_t c = rel.atIneq64(r1, localOffset + j);
        if (c != 0) {
          num =
              isl_aff_add(num, isl_aff_scale_val(isl_aff_copy(localAffs[j]),
                                                 isl_val_int_from_si(ctx, c)));
        }
      }
      localAffs[k] = isl_aff_floor(
          isl_aff_scale_down_val(num, isl_val_int_from_si(ctx, d)));
    }
    isl_multi_aff* maff = isl_multi_aff_zero(
        isl_space_alloc(ctx, numSymbols, numDimVars, fullSetDim));
    for (unsigned j = 0; j < numDimVars; ++j) {
      maff = isl_multi_aff_set_aff(
          maff, j,
          isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, j));
    }
    for (unsigned k = 0; k < numLocals; ++k) {
      maff = isl_multi_aff_set_aff(maff, numDimVars + k, localAffs[k]);
    }
    isl_local_space_free(ls);

    isl_space* fullSetSpace = isl_space_set_alloc(ctx, numSymbols, fullSetDim);
    isl_basic_set* bset = isl_basic_set_from_constraint_matrices(
        fullSetSpace, eqMat, ineqMat, isl_dim_set, isl_dim_div, isl_dim_param,
        isl_dim_cst);
    bset = isl_basic_set_preimage_multi_aff(bset, maff);
    isl_basic_map* res = isl_basic_map_from_range(bset);
    return isl_basic_map_move_dims(res, isl_dim_in, 0, isl_dim_out, 0,
                                   numDomain);
  }

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
