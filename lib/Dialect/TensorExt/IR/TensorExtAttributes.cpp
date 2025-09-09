#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "lib/Utils/Layout/IslConversion.h"
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
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

LogicalResult AlignmentAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, mlir::DenseI64ArrayAttr in,
    mlir::DenseI64ArrayAttr out, mlir::DenseI64ArrayAttr insertedDims,
    mlir::DenseI64ArrayAttr padding, TypedAttr paddingValue) {
  if (out.empty()) {
    return emitError() << "out may not be an empty array";
  }

  if (in.size() + insertedDims.size() != out.size()) {
    return emitError()
           << "in.size() + insertedDims.size() must equal out.size()";
  }

  for (auto dim : insertedDims.asArrayRef()) {
    if (dim < 0 || dim >= out.size()) {
      return emitError() << "insertedDims must be in the range [0, out.size())";
    }
  }

  for (int i = 0; i < out.size(); i++) {
    if (out[i] <= 0) {
      return emitError() << "out dimension " << i
                         << " must be positive, but was " << out[i];
    }
  }

  for (int i = 0; i < in.size(); i++) {
    if (in[i] <= 0) {
      return emitError() << "in dimension " << i
                         << " must be positive, but was " << in[i];
    }
  }

  if (!padding.empty() && padding.size() != out.size()) {
    return emitError() << "padding.size() must equal out.size()";
  }

  if (!padding.empty() && !paddingValue) {
    return emitError() << "paddingValue must be set if padding is set";
  }

  DenseSet<int64_t> insertedDimsSet(insertedDims.asArrayRef().begin(),
                                    insertedDims.asArrayRef().end());
  if (insertedDimsSet.size() != insertedDims.size()) {
    return emitError() << "insertedDims must all be unique";
  }

  // Rewrite the tensor shape with expanded dims and padding
  SmallVector<int64_t> beforeReplication;
  beforeReplication.resize(out.size(), 1);
  int inIndex = 0;
  for (int i = 0; i < out.size(); i++) {
    if (!insertedDimsSet.count(i)) {
      beforeReplication[i] = in[inIndex++];
    }
  }

  if (!padding.empty()) {
    for (int i = 0; i < out.size(); i++) {
      beforeReplication[i] += padding[i];
    }
  }

  // For each axis, input dim + padding divides or is divisible by output dim,
  // which enables replication along each axis.
  for (int i = 0; i < out.size(); i++) {
    if (beforeReplication[i] % out[i] != 0 &&
        (beforeReplication[i] == 0 || out[i] % beforeReplication[i] != 0)) {
      std::string str;
      llvm::raw_string_ostream os(str);
      os << "After inserting dims and padding, each axis must have size "
            "dividing or divisible by the corresponding output axis size, but "
            "found size=";
      llvm::interleaveComma(beforeReplication, os);
      os << " and out=";
      llvm::interleaveComma(out.asArrayRef(), os);
      return emitError() << os.str();
    }
  }

  return success();
}

LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 AffineMap map, AlignmentAttr alignment) {
  if (alignment && map.getNumDims() != alignment.getOut().size()) {
    return emitError() << "The affine map's input size (" << map.getNumDims()
                       << ") must match the "
                          "number of dimensions of alignment.out ("
                       << alignment.getOut().size() << ")";
  }

  return success();
}

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
