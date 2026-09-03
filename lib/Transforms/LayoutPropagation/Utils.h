#ifndef LIB_TRANSFORMS_LAYOUTPROPAGATION_UTILS_H_
#define LIB_TRANSFORMS_LAYOUTPROPAGATION_UTILS_H_

#include <cstdint>
#include <optional>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {

constexpr StringLiteral kKernelInfoAttrName = "heir.kernel_info";
constexpr StringLiteral kKernelInputShapeKey = "input_shape";
constexpr StringLiteral kKernelShapeKey = "result_shape";
constexpr StringLiteral kGapFactorKey = "gap_factor";

// Symmetric zero padding on the spatial dims that LayoutPropagation folded out
// of a `tensor.pad` and into a conv's own `padding` parameter.
constexpr StringLiteral kConvFoldedPaddingAttrName = "heir.conv_folded_padding";

// The width of the matrix that a conv's diagonalized filter layout was built
// against, when LayoutPropagation absorbed the data operand's slot packing into
// the filter's column space. The columns are then ciphertext slots, so the
// layout is built at the ciphertext size instead of the expanded Toeplitz
// matrix's own C*W. Absent means the layout uses the Toeplitz width.
constexpr StringLiteral kAbsorbedMatrixWidthAttrName =
    "heir.absorbed_matrix_width";

// A conv's data operand as its expanded Toeplitz matrix sees it, paired with
// the conv `padding` parameter that goes with it.
struct ConvMatrixOperand {
  RankedTensorType dataType;
  int64_t padding = 0;
};

// Removes `padding` entries from both ends of every spatial dim of a conv data
// operand: the width dim of a rank-3 (N, C, W) operand, or the height and
// width dims of a rank-4 (N, C, H, W) one. A conv's `padding` parameter is a
// single symmetric value shared by every spatial dim, which is why one
// `padding` covers them all.
std::optional<ConvMatrixOperand> foldConvSpatialPadding(
    RankedTensorType dataType, int64_t padding);

// Reads back an integer conv kernel parameter LayoutPropagation recorded on
// `op`; 0 when the attribute is absent.
int64_t getConvKernelParam(Operation* op, StringRef name);

// Records an integer conv kernel parameter on `op`. A value of 0 removes the
// attribute instead of writing it, so an op clone cannot carry a stale one.
void setConvKernelParam(Operation* op, StringRef name, int64_t value);

// The padding folded into `op`'s own padding parameter; 0 if none.
inline int64_t getConvFoldedPadding(Operation* op) {
  return getConvKernelParam(op, kConvFoldedPaddingAttrName);
}
inline void setConvFoldedPadding(Operation* op, int64_t padding) {
  setConvKernelParam(op, kConvFoldedPaddingAttrName, padding);
}

// The width `op`'s filter layout was diagonalized at; 0 when it absorbed
// nothing and so uses the expanded Toeplitz width.
inline int64_t getAbsorbedMatrixWidth(Operation* op) {
  return getConvKernelParam(op, kAbsorbedMatrixWidthAttrName);
}
inline void setAbsorbedMatrixWidth(Operation* op, int64_t width) {
  setConvKernelParam(op, kAbsorbedMatrixWidthAttrName, width);
}

struct KernelInfo {
  SmallVector<int64_t> inputShape;
  // Tracks the shape of the resolved kernel's tensor shape. This will account
  // for any additional expansion or striding due to the FHE kernel.
  SmallVector<int64_t> resultShape;
  // Tracks the gap factor used for multiplexing convolutions.
  int64_t gapFactor = 1;
};

Attribute makeKernelInfoAttr(MLIRContext* ctx, const KernelInfo& info);

std::optional<KernelInfo> getKernelInfo(Attribute attr);

using tensor_ext::LayoutAttr;

int64_t maxOfMaxes(::llvm::ArrayRef<int64_t> d1, ::llvm::ArrayRef<int64_t> d2);

// `dims` is a list of dims of a tensor, and `inserts` represents a list of
// inserted dims in that tensor (via alignment attr's insertedDims). This
// function shifts the values of `dims` as if the dims from `inserts` were
// inserted.
//
// Example:
//
// input = [0, 1, 2, 3]
// inserts = [1, 2]
//
// output = [0, 3, 4, 5]
//
// This allows you to track how `dims` maps to new dims as a result of the
// inserts.
::llvm::SmallVector<int64_t> shiftByInserted(::llvm::ArrayRef<int64_t> dims,
                                             ::llvm::ArrayRef<int64_t> inserts,
                                             bool increment = true);

// Map the dims in the `dims` list to new dims when the dims in `removed` are
// removed from the tensor.
//
// Assumes dims and removed do not have any common values
//
// Example:
// input = [0, 3, 4, 5]
// removed = [1, 2]
//
// output = [0, 1, 2, 3]
::llvm::SmallVector<int64_t> shiftByRemoved(::llvm::ArrayRef<int64_t> dims,
                                            ::llvm::ArrayRef<int64_t> removed);

// A helper to convert the layout of an input tensor to a reduce op. The result
// layout is equivalent to setting the summed dimensions to 0.
LayoutAttr convertLayoutForReduce(LayoutAttr inputLayout,
                                  ArrayRef<int64_t> dimsToReduce);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTPROPAGATION_UTILS_H_
