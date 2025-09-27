#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"

#include <cassert>
#include <cstdlib>
#include <string>

#include "lib/Utils/Layout/IslConversion.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"        // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"         // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

// ISL
#include "include/isl/ctx.h"         // from @isl
#include "include/isl/map.h"         // from @isl
#include "include/isl/map_type.h"    // from @isl
#include "include/isl/space_type.h"  // from @isl

namespace mlir {
namespace heir {
namespace tensor_ext {

using presburger::IntegerPolyhedron;
using presburger::IntegerRelation;
using presburger::VarKind;

LogicalResult NewLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr layoutStr) {
  auto result = getIntegerRelationFromIslStr(layoutStr.getValue().str());
  if (failed(result)) {
    return emitError() << "Failed to parse the layout string (ISL): "
                       << layoutStr;
  }
  // Success if you can parse the ISL string and convert it.
  return success();
}

IntegerRelation NewLayoutAttr::getIntegerRelation() const {
  auto result = getIntegerRelationFromIslStr(getLayoutStr());
  assert(succeeded(result) && "Failed to parse the layout string");
  return result.value();
}

NewLayoutAttr NewLayoutAttr::getFromIntegerRelation(
    ::mlir::MLIRContext* context, const IntegerRelation& relation) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* bmap = convertRelationToBasicMap(relation, ctx);

  bmap = isl_basic_map_set_dim_name(bmap, isl_dim_out, 0, "ct");
  bmap = isl_basic_map_set_dim_name(bmap, isl_dim_out, 1, "slot");

  char* resultStr = isl_basic_map_to_str(bmap);
  std::string layoutStr(resultStr);
  free(resultStr);
  isl_basic_map_free(bmap);
  isl_ctx_free(ctx);
  return NewLayoutAttr::get(context, layoutStr);
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
