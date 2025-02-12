#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"

#include <cstdint>
#include <string>

#include "lib/Utils/AffineMapUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"     // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

// Kept inside a namespace because it generates a function called
// populateWithGenerated, which can conflict with other generated patterns.
#include "lib/Dialect/TensorExt/IR/TensorExtCanonicalization.cpp.inc"

void RotateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  populateWithGenerated(results);
}

LogicalResult RotateOp::verify() {
  auto x = getTensor().getType();
  // TODO(#924): Currently RotateOp only supports rotating a 1-D vector, or a
  // vector with only one non-unit dimension that is treated as the major
  // dimension.
  if (x.getRank() != 1) {
    if (llvm::count_if(x.getShape(), [](auto dim) { return dim != 1; }) != 1) {
      return emitOpError() << "requires a 1-D input tensor or tensor with "
                              "single non-unit dimension, but found "
                           << x;
    }
  }
  return success();
}

LogicalResult verifyLayoutMatchesType(const AffineMap &layout, Type type,
                                      Operation *op) {
  int64_t rank = cast<ShapedType>(type).getRank();
  if (rank != layout.getNumDims()) {
    std::string layoutStr;
    llvm::raw_string_ostream os(layoutStr);
    layout.print(os);

    return op->emitOpError()
           << "requires tensor rank to match the layout map's dimension count"
              " but found rank "
           << rank << " and map " << os.str();
  }

  return success();
}

LogicalResult ConvertLayoutOp::verify() {
  LogicalResult inputVerification = verifyLayoutMatchesType(
      getFromLayout().getValue(), getTensor().getType(), *this);
  if (failed(inputVerification)) {
    return inputVerification;
  }

  LogicalResult outputVerification = verifyLayoutMatchesType(
      getToLayout().getValue(), getResult().getType(), *this);
  if (failed(outputVerification)) {
    return outputVerification;
  }

  return success();
}

LogicalResult AssignLayoutOp::verify() {
  return verifyLayoutMatchesType(getLayout().getValue(), getTensor().getType(),
                                 *this);
}

LogicalResult PermuteOp::verify() {
  auto tensorTy = getInput().getType();
  // TODO(#924): Support more general vector inputs.
  if (tensorTy.getRank() != 1) {
    return emitOpError() << "requires a 1-D input tensor"
                            "single non-unit dimension, but found "
                         << tensorTy;
  }

  // Assert it's a permutation; would that be too slow for a verifier?
  auto affineMapAttr = dyn_cast<AffineMapAttr>(getPermutation());
  SmallVector<int64_t> permutationResult;
  if (affineMapAttr) {
    if (failed(makeExplicit1DMapping(affineMapAttr.getValue(),
                                     tensorTy.getNumElements(),
                                     permutationResult))) {
      return emitOpError()
             << "failed to materialize affine map attr as permutation";
    }

    if (!isPermutation(permutationResult)) {
      return emitOpError() << "expected permutation, but got an affine_map "
                              "attr that was not a permutation.";
    }
  } else {
    auto denseElementsAttr = dyn_cast<DenseIntElementsAttr>(getPermutation());
    if (denseElementsAttr) {
      permutationResult = llvm::map_to_vector(
          denseElementsAttr, [](const APInt &i) { return i.getSExtValue(); });
      if (!isPermutation(permutationResult)) {
        return emitOpError() << "expected permutation, but got a dense "
                                "attr that was not a permutation.";
      }
    } else {
      return emitOpError() << "unknown permutation type";
    }
  }

  return success();
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
