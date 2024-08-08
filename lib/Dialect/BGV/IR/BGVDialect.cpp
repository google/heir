#include "lib/Dialect/BGV/IR/BGVDialect.h"

#include <optional>

#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/FHEHelpers.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

// Generated definitions
#include "lib/Dialect/BGV/IR/BGVDialect.cpp.inc"
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#define GET_OP_CLASSES
#include "lib/Dialect/BGV/IR/BGVOps.cpp.inc"

namespace mlir {
namespace heir {
namespace bgv {

//===----------------------------------------------------------------------===//
// BGV dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void BGVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/BGV/IR/BGVOps.cpp.inc"
      >();
}

// TODO: re-enable verifiers with RNS support
// LogicalResult MulOp::verify() {
//   auto x = getLhs().getType();
//   auto y = getRhs().getType();
//   if (x.getRlweParams().getDimension() != y.getRlweParams().getDimension()) {
//     return emitOpError() << "input dimensions do not match";
//   }
//   auto out = getOutput().getType();
//   if (out.getRlweParams().getDimension() !=
//       y.getRlweParams().getDimension() + x.getRlweParams().getDimension() -
//       1) {
//     return emitOpError() << "output.dim == x.dim + y.dim - 1 does not hold";
//   }
//   return success();
// }
//
// LogicalResult RotateOp::verify() {
//   auto x = getInput().getType();
//   if (x.getRlweParams().getDimension() != 2) {
//     return emitOpError() << "x.dim == 2 does not hold";
//   }
//   auto out = getOutput().getType();
//   if (out.getRlweParams().getDimension() != 2) {
//     return emitOpError() << "output.dim == 2 does not hold";
//   }
//   return success();
// }
//
// LogicalResult RelinearizeOp::verify() {
//   auto x = getInput().getType();
//   auto out = getOutput().getType();
//   if (x.getRlweParams().getDimension() != getFromBasis().size()) {
//     return emitOpError() << "input dimension does not match from_basis";
//   }
//   if (out.getRlweParams().getDimension() != getToBasis().size()) {
//     return emitOpError() << "output dimension does not match to_basis";
//   }
//   return success();
// }
//
// LogicalResult ModulusSwitchOp::verify() {
//   auto x = getInput().getType();
//   auto xRing = x.getRlweParams().getRing();
//
//   auto out = getOutput().getType();
//   auto outRing = out.getRlweParams().getRing();
//   if (outRing != getToRing()) {
//     return emitOpError() << "output ring should match to_ring";
//   }
//   if (xRing.getCoefficientModulus().getValue().ule(
//           outRing.getCoefficientModulus().getValue())) {
//     return emitOpError()
//            << "output ring modulus should be less than the input ring
//            modulus";
//   }
//   if (!xRing.getCoefficientModulus()
//            .getValue()
//            .urem(outRing.getCoefficientModulus().getValue())
//            .isZero()) {
//     return emitOpError()
//            << "output ring modulus should divide the input ring modulus";
//   }
//
//   return success();
// }

LogicalResult MulOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, MulOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (auto x = dyn_cast<lwe::RLWECiphertextType>(adaptor.getLhs().getType())) {
    auto y = cast<lwe::RLWECiphertextType>(adaptor.getRhs().getType());
    auto newDim =
        x.getRlweParams().getDimension() + y.getRlweParams().getDimension() - 1;
    inferredReturnTypes.push_back(lwe::RLWECiphertextType::get(
        ctx, x.getEncoding(),
        lwe::RLWEParamsAttr::get(ctx, newDim, x.getRlweParams().getRing()),
        x.getUnderlyingType()));
    return success();
  }
  if (auto xr =
          dyn_cast<lwe::RNSRLWECiphertextType>(adaptor.getLhs().getType())) {
    auto yr = cast<lwe::RNSRLWECiphertextType>(adaptor.getRhs().getType());
    auto newDim = xr.getRlweParams().getDimension() +
                  yr.getRlweParams().getDimension() - 1;
    inferredReturnTypes.push_back(lwe::RNSRLWECiphertextType::get(
        ctx, xr.getEncoding(),
        lwe::RNSRLWEParamsAttr::get(ctx, newDim, xr.getRlweParams().getRing()),
        xr.getUnderlyingType()));
    return success();
  }
  return failure();
}

LogicalResult RelinearizeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, RelinearizeOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (auto x =
          dyn_cast<lwe::RLWECiphertextType>(adaptor.getInput().getType())) {
    inferredReturnTypes.push_back(lwe::RLWECiphertextType::get(
        ctx, x.getEncoding(),
        lwe::RLWEParamsAttr::get(ctx, 2, x.getRlweParams().getRing()),
        x.getUnderlyingType()));
    return success();
  }
  if (auto xr =
          dyn_cast<lwe::RNSRLWECiphertextType>(adaptor.getInput().getType())) {
    inferredReturnTypes.push_back(lwe::RNSRLWECiphertextType::get(
        ctx, xr.getEncoding(),
        lwe::RNSRLWEParamsAttr::get(ctx, 2, xr.getRlweParams().getRing()),
        xr.getUnderlyingType()));
    return success();
  }
  return failure();
}

// LogicalResult ExtractOp::verify() {
//   auto inputTy = getInput().getType();
//   auto tensorTy = dyn_cast<RankedTensorType>(inputTy.getUnderlyingType());
//   if (!tensorTy) {
//     return emitOpError() << "input RLWE ciphertext type must have a ranked "
//                             "tensor as its underlying_type, but found "
//                          << inputTy.getUnderlyingType();
//   }
//
//   auto outputScalarType = getOutput().getType().getUnderlyingType();
//   if (tensorTy.getElementType() != outputScalarType) {
//     return emitOpError() << "output RLWE ciphertext's underlying_type must be
//     "
//                             "the element type of the input ciphertext's "
//                             "underlying tensor type, but found tensor type "
//                          << tensorTy << " and output type " <<
//                          outputScalarType;
//   }
//   return success();
// }

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
