#include "lib/Dialect/LWE/IR/LWEAttributes.h"

#include <cstdint>

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/APIntUtils.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

#define DEBUG_TYPE "lwe-attributes"

namespace mlir {
namespace heir {
namespace lwe {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

int64_t getScalingFactorFromEncodingAttr(Attribute encoding) {
  return llvm::TypeSwitch<Attribute, int64_t>(encoding)
      .Case<FullCRTPackingEncodingAttr>(
          [](auto attr) { return attr.getScalingFactor(); })
      .Case<InverseCanonicalEncodingAttr>(
          [](auto attr) { return attr.getScalingFactor(); })
      .Default([](Attribute) { return 0; });
}

int64_t inferMulOpScalingFactor(Attribute xEncoding, Attribute yEncoding,
                                int64_t plaintextModulus) {
  int64_t xScale = getScalingFactorFromEncodingAttr(xEncoding);
  int64_t yScale = getScalingFactorFromEncodingAttr(yEncoding);
  return llvm::TypeSwitch<Attribute, int64_t>(xEncoding)
      .Case<FullCRTPackingEncodingAttr>(
          // Use 128-bit int in case of large ptm.
          [&](auto attr) {
            return (APInt(128, xScale) * APInt(128, yScale))
                .urem(plaintextModulus);
          })
      .Case<InverseCanonicalEncodingAttr>(
          [&](auto attr) { return xScale + yScale; })
      .Default([](Attribute) { return 0; });
}

int64_t inferModulusSwitchOrRescaleOpScalingFactor(Attribute xEncoding,
                                                   APInt dividedModulus,
                                                   int64_t plaintextModulus) {
  int64_t xScale = getScalingFactorFromEncodingAttr(xEncoding);
  return llvm::TypeSwitch<Attribute, int64_t>(xEncoding)
      .Case<FullCRTPackingEncodingAttr>([&](auto attr) {
        // Use 128-bit int in case of large ptm.
        auto qInvT = multiplicativeInverse(
            APInt(128, dividedModulus.urem(plaintextModulus)),
            APInt(128, plaintextModulus));
        return (APInt(128, xScale) * qInvT).urem(plaintextModulus);
      })
      .Case<InverseCanonicalEncodingAttr>([&](auto attr) {
        // skip if xScale is 0
        if (xScale == 0) return xScale;
        // round to nearest log2 instead of ceil
        auto logQ = dividedModulus.nearestLogBase2();
        LLVM_DEBUG(llvm::dbgs() << "inferring new scale; logQ=" << logQ
                                << ", xScale=" << xScale << "\n");
        return xScale - logQ;
      })
      .Default([](Attribute) { return 0; });
}

Attribute getEncodingAttrWithNewScalingFactor(Attribute encoding,
                                              int64_t newScale) {
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
  LLVM_DEBUG(llvm::dbgs() << "dividedModulus=" << dividedModulus
                          << " new scale=" << newScale << "\n");
  return PlaintextSpaceAttr::get(
      ctx, xRing, getEncodingAttrWithNewScalingFactor(xEncoding, newScale));
}

polynomial::RingAttr getRlweRNSRingWithLevel(polynomial::RingAttr ringAttr,
                                             int level) {
  auto rnsType = cast<rns::RNSType>(ringAttr.getCoefficientType());
  auto newRnsType = rns::RNSType::get(
      rnsType.getContext(), rnsType.getBasisTypes().take_front(level + 1));
  return polynomial::RingAttr::get(newRnsType, ringAttr.getPolynomialModulus());
}

polynomial::RingAttr getRingFromModulusChain(
    ModulusChainAttr chainAttr,
    polynomial::IntPolynomialAttr polynomialModulus) {
  SmallVector<Type> limbTypes = llvm::to_vector(llvm::map_range(
      chainAttr.getElements(), [](mlir::IntegerAttr attr) -> Type {
        return mod_arith::ModArithType::get(attr.getType().getContext(), attr);
      }));
  rns::RNSType rnsType = rns::RNSType::get(
      chainAttr.getContext(),
      ArrayRef<Type>(limbTypes).take_front(chainAttr.getCurrent() + 1));
  return polynomial::RingAttr::get(rnsType, polynomialModulus);
}

//===----------------------------------------------------------------------===//
// Attribute Verification
//===----------------------------------------------------------------------===//

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
