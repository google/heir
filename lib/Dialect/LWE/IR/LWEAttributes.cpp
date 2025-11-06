#include "lib/Dialect/LWE/IR/LWEAttributes.h"

#include <cstdint>

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Utils/APIntUtils.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

llvm::APInt getScalingFactorFromEncodingAttr(Attribute encoding) {
  return llvm::TypeSwitch<Attribute, llvm::APInt>(encoding)
      .Case<FullCRTPackingEncodingAttr>(
          [](auto attr) { return attr.getScalingFactor(); })
      .Case<InverseCanonicalEncodingAttr>(
          [](auto attr) { return attr.getScalingFactor(); })
      .Default([](Attribute) { return llvm::APInt(64, 0); });
}

llvm::APInt inferMulOpScalingFactor(Attribute xEncoding, Attribute yEncoding,
                                    int64_t plaintextModulus) {
  llvm::APInt xScale = getScalingFactorFromEncodingAttr(xEncoding);
  llvm::APInt yScale = getScalingFactorFromEncodingAttr(yEncoding);
  return llvm::TypeSwitch<Attribute, llvm::APInt>(xEncoding)
      .Case<FullCRTPackingEncodingAttr>(
          // Use 128-bit int in case of large ptm.
          [&](auto attr) {
            llvm::APInt xScale128 = xScale.zext(128);
            llvm::APInt yScale128 = yScale.zext(128);
            llvm::APInt result = (xScale128 * yScale128)
                                     .urem(llvm::APInt(128, plaintextModulus));
            // Reduce to minimum bit width for consistent comparisons
            unsigned minBits = result.getActiveBits();
            if (minBits == 0) minBits = 1;  // APInt needs at least 1 bit
            return result.trunc(std::max(64u, minBits));
          })
      .Case<InverseCanonicalEncodingAttr>([&](auto attr) {
        // Ensure both scales have the same bit width before adding
        unsigned maxBitWidth =
            std::max(xScale.getBitWidth(), yScale.getBitWidth());
        llvm::APInt xScaleExt = xScale.zext(maxBitWidth);
        llvm::APInt yScaleExt = yScale.zext(maxBitWidth);
        return xScaleExt + yScaleExt;
      })
      .Default([](Attribute) { return llvm::APInt(64, 0); });
}

llvm::APInt inferModulusSwitchOrRescaleOpScalingFactor(
    Attribute xEncoding, APInt dividedModulus, int64_t plaintextModulus) {
  llvm::APInt xScale = getScalingFactorFromEncodingAttr(xEncoding);
  return llvm::TypeSwitch<Attribute, llvm::APInt>(xEncoding)
      .Case<FullCRTPackingEncodingAttr>([&](auto attr) {
        // Use 128-bit int in case of large ptm.
        auto qInvT = multiplicativeInverse(
            APInt(128, dividedModulus.urem(plaintextModulus)),
            APInt(128, plaintextModulus));
        llvm::APInt xScale128 = xScale.zext(128);
        llvm::APInt result =
            (xScale128 * qInvT).urem(llvm::APInt(128, plaintextModulus));
        // Reduce to minimum bit width for consistent comparisons
        unsigned minBits = result.getActiveBits();
        if (minBits == 0) minBits = 1;  // APInt needs at least 1 bit
        return result.trunc(std::max(64u, minBits));
      })
      .Case<InverseCanonicalEncodingAttr>([&](auto attr) {
        // skip if xScale is 0
        if (xScale.isZero()) return xScale;
        // round to nearest log2 instead of ceil
        auto logQ = dividedModulus.nearestLogBase2();
        return xScale - llvm::APInt(xScale.getBitWidth(), logQ);
      })
      .Default([](Attribute) { return llvm::APInt(64, 0); });
}

Attribute getEncodingAttrWithNewScalingFactor(Attribute encoding,
                                              const llvm::APInt& newScale) {
  return llvm::TypeSwitch<Attribute, Attribute>(encoding)
      .Case<FullCRTPackingEncodingAttr>([&](auto attr) {
        return FullCRTPackingEncodingAttr::get(encoding.getContext(), newScale);
      })
      .Case<InverseCanonicalEncodingAttr>([&](auto attr) {
        return InverseCanonicalEncodingAttr::get(encoding.getContext(),
                                                 newScale);
      })
      .Default([](Attribute) { return nullptr; });
}

PlaintextSpaceAttr inferMulOpPlaintextSpaceAttr(MLIRContext* ctx,
                                                PlaintextSpaceAttr x,
                                                PlaintextSpaceAttr y) {
  auto xRing = x.getRing();
  auto xEncoding = x.getEncoding();
  auto yEncoding = y.getEncoding();

  int64_t plaintextModulus = 0;
  if (auto modArithType =
          llvm::dyn_cast<mod_arith::ModArithType>(xRing.getCoefficientType())) {
    plaintextModulus = modArithType.getModulus().getValue().getSExtValue();
  }

  auto newScale =
      inferMulOpScalingFactor(xEncoding, yEncoding, plaintextModulus);
  return PlaintextSpaceAttr::get(
      ctx, xRing, getEncodingAttrWithNewScalingFactor(xEncoding, newScale));
}

PlaintextSpaceAttr inferModulusSwitchOrRescaleOpPlaintextSpaceAttr(
    MLIRContext* ctx, PlaintextSpaceAttr x, APInt dividedModulus) {
  auto xRing = x.getRing();
  auto xEncoding = x.getEncoding();

  int64_t plaintextModulus = 0;
  if (auto modArithType =
          llvm::dyn_cast<mod_arith::ModArithType>(xRing.getCoefficientType())) {
    plaintextModulus = modArithType.getModulus().getValue().getSExtValue();
  }

  auto newScale = inferModulusSwitchOrRescaleOpScalingFactor(
      xEncoding, dividedModulus, plaintextModulus);
  return PlaintextSpaceAttr::get(
      ctx, xRing, getEncodingAttrWithNewScalingFactor(xEncoding, newScale));
}

//===----------------------------------------------------------------------===//
// Attribute Verification
//===----------------------------------------------------------------------===//

LogicalResult ApplicationDataAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::Type messageType, Attribute overflow) {
  if (!mlir::isa<PreserveOverflowAttr, NoOverflowAttr>(overflow)) {
    return emitError() << "overflow must be either preserve_overflow or "
                       << "no_overflow, but found " << overflow << "\n";
  }

  return success();
}

LogicalResult PlaintextSpaceAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::heir::polynomial::RingAttr ring, Attribute encoding) {
  if (mlir::isa<FullCRTPackingEncodingAttr>(encoding)) {
    // For full CRT packing, the ring must be of the form x^n + 1 and the
    // modulus must be 1 mod n.
    auto polyMod = ring.getPolynomialModulus();
    auto poly = polyMod.getPolynomial();
    auto polyTerms = poly.getTerms();
    if (polyTerms.size() != 2) {
      return emitError() << "polynomial modulus must be of the form x^n + 1, "
                         << "but found " << polyMod << "\n";
    }
    const auto& constantTerm = polyTerms[0];
    const auto& constantCoeff = constantTerm.getCoefficient();
    if (!(constantTerm.getExponent().isZero() && constantCoeff.isOne() &&
          polyTerms[1].getCoefficient().isOne())) {
      return emitError() << "polynomial modulus must be of the form x^n + 1, "
                         << "but found " << polyMod << "\n";
    }
    // Check that the modulus is 1 mod n.
    auto modCoeffTy =
        llvm::dyn_cast<mod_arith::ModArithType>(ring.getCoefficientType());
    if (modCoeffTy) {
      APInt modulus = modCoeffTy.getModulus().getValue();
      unsigned n = poly.getDegree();
      if (!modulus.urem(APInt(modulus.getBitWidth(), n)).isOne()) {
        return emitError()
               << "modulus must be 1 mod n for full CRT packing, mod = "
               << modulus.getZExtValue() << " n = " << n << "\n";
      }
    }
  }

  return success();
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
