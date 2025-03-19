#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"

#include "llvm/include/llvm/ADT/STLExtras.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

LogicalResult AlignmentAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::detail::DenseArrayAttrImpl<long> in,
    mlir::detail::DenseArrayAttrImpl<long> out,
    mlir::detail::DenseArrayAttrImpl<long> insertedDims,
    mlir::detail::DenseArrayAttrImpl<long> padding, TypedAttr paddingValue) {
  if (in.empty() || out.empty()) {
    return emitError() << "in and out may not be empty arrays";
  }

  if (in.size() + insertedDims.size() != out.size()) {
    return emitError()
           << "in.size() + insertedDims.size() must equal out.size()";
  }

  if (!padding.empty() && padding.size() != out.size()) {
    return emitError() << "padding.size() must equal out.size()";
  }

  if (!padding.empty() && !paddingValue) {
    return emitError() << "paddingValue must be set if padding is set";
  }

  DenseSet<long> insertedDimsSet(insertedDims.asArrayRef().begin(),
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

  // For each axis, input dim + padding divides or is divisibile by output dim,
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

LogicalResult LayoutAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::AffineMap map, AlignmentAttr alignment) {
  if (alignment && map.getNumDims() != alignment.getOut().size()) {
    return emitError() << "The affine map's input size must match the "
                          "number of dimensions of alignment.out";
  }

  return success();
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
