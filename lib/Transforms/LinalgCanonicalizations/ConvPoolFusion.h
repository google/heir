#ifndef LIB_TRANSFORMS_LINALGCANONICALIZATIONS_CONVPOOLFUSION_H_
#define LIB_TRANSFORMS_LINALGCANONICALIZATIONS_CONVPOOLFUSION_H_

#include <cstdint>
#include <optional>

#include "llvm/include/llvm/ADT/APFloat.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/APInt.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {

// Spatial, stride, and channel parameters for the 2D convolution layer.
struct Conv2DSizeParams {
  int64_t F;          // Number of output channels (filters)
  int64_t C;          // Number of input channels
  int64_t kH;         // Kernel height
  int64_t kW;         // Kernel width
  int64_t strideH;    // Stride height
  int64_t strideW;    // Stride width
  int64_t dilationH;  // Dilation height
  int64_t dilationW;  // Dilation width
};

// Window, stride, and dilation parameters for the 2D pooling layer.
struct Pool2DSizeParams {
  int64_t pH;         // Pool window height
  int64_t pW;         // Pool window width
  int64_t strideH;    // Stride height
  int64_t strideW;    // Stride width
  int64_t dilationH;  // Dilation height
  int64_t dilationW;  // Dilation width
};

// Final spatial dimensions of the pooling layer's output.
// Used as the starting point for bottom-up input shape generation.
struct OutputSizeParams {
  int64_t poolOutH;  // Pooling output height
  int64_t poolOutW;  // Pooling output width
};

int64_t getEffectiveKernelHeight(const Conv2DSizeParams& conv,
                                 const Pool2DSizeParams& pool);
int64_t getEffectiveKernelWidth(const Conv2DSizeParams& conv,
                                const Pool2DSizeParams& pool);

struct ScalingFactorInfo {
  llvm::APFloat scale;
  Operation* op;
};

// Fuses a convolution filter with a subsequent pooling and rescale operation.
// See conv_pool_proof.tex,pdf for an explanation of the equivalence.
llvm::SmallVector<llvm::APFloat> fuseFloatFilters(
    llvm::ArrayRef<llvm::APFloat> convValues, const Conv2DSizeParams& conv,
    const Pool2DSizeParams& pool, const llvm::fltSemantics& semantics,
    std::optional<llvm::APFloat> scaleVal);

// Fuses a convolution filter with a subsequent pooling and rescale operation.
// See conv_pool_proof.tex,pdf for an explanation of the equivalence.
llvm::SmallVector<llvm::APInt> fuseIntFilters(
    llvm::ArrayRef<llvm::APInt> convValues, const Conv2DSizeParams& conv,
    const Pool2DSizeParams& pool, unsigned width);

// MLIR-dependent wrappers
DenseElementsAttr fuseConv2DPoolingFilters(
    DenseElementsAttr convFilter, const Conv2DSizeParams& conv,
    const Pool2DSizeParams& pool, std::optional<llvm::APFloat> scaleVal);

FailureOr<ScalingFactorInfo> matchScalingFactor(
    Operation* user, Value poolResult, const llvm::fltSemantics& semantics);

std::optional<llvm::APFloat> extractFloatAttr(TypedAttr attr);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LINALGCANONICALIZATIONS_CONVPOOLFUSION_H_
