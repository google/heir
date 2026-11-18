#include "lib/Dialect/LWE/IR/LWEAttributes.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
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

Attribute inferMulOpEncoding(Attribute x, Attribute y) {
  if (auto xIc = llvm::dyn_cast<InverseCanonicalEncodingAttr>(x)) {
    int64_t xScale = xIc.getScalingFactor();
    int64_t yScale = getScalingFactorFromEncodingAttr(y);
    int64_t newScale = (xScale == -1 || yScale == -1) ? -1 : xScale + yScale;
    return InverseCanonicalEncodingAttr::get(x.getContext(), newScale);
  }
  return x;
}

Attribute inferModulusSwitchOrRescaleOpEncoding(Attribute x,
                                                const APInt& dividedModulus) {
  if (auto xIc = llvm::dyn_cast<InverseCanonicalEncodingAttr>(x)) {
    int64_t xScale = xIc.getScalingFactor();
    // Skip if xScale is unset (-1) or 0.
    if (xScale == -1 || xScale == 0) return x;
    // Round to nearest log2 using floating point when width fits in double.
    // This matches the behavior in CKKSScaleModel evalModReduceScale and
    // SchemeParams.
    int64_t logQ;
    if (dividedModulus.getActiveBits() <= 64) {
      logQ = static_cast<int64_t>(
          std::llround(std::log2(dividedModulus.roundToDouble())));
    } else {
      logQ = dividedModulus.nearestLogBase2();
    }
    LLVM_DEBUG(llvm::dbgs() << "inferring new scale; logQ=" << logQ
                            << ", xScale=" << xScale << "\n");
    return InverseCanonicalEncodingAttr::get(x.getContext(), xScale - logQ);
  }
  return x;
}

bool areEncodingsCompatible(Attribute a, Attribute b) {
  if (!a || !b) return a == b;
  if (llvm::isa<FullCRTPackingEncodingAttr>(a) &&
      llvm::isa<FullCRTPackingEncodingAttr>(b)) {
    return true;
  }
  if (auto aCC = llvm::dyn_cast<ConstantCoefficientEncodingAttr>(a)) {
    if (auto bCC = llvm::dyn_cast<ConstantCoefficientEncodingAttr>(b)) {
      return aCC.getScalingFactor() == bCC.getScalingFactor() &&
             (aCC.getPlaintextBits() == bCC.getPlaintextBits() ||
              aCC.getPlaintextBits() == 0 || bCC.getPlaintextBits() == 0);
    }
    return false;
  }
  return a == b;
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

polynomial::RingAttr getRlweRNSRingWithLevel(polynomial::RingAttr ringAttr,
                                             ArrayRef<int32_t> moduliPerLevel,
                                             int level) {
  auto rnsType = cast<rns::RNSType>(ringAttr.getCoefficientType());
  int32_t numModuli =
      moduliPerLevel.empty()
          ? level + 1
          : std::accumulate(moduliPerLevel.begin(),
                            moduliPerLevel.begin() + level + 1, 0);
  auto newRnsType = rns::RNSType::get(
      rnsType.getContext(), rnsType.getBasisTypes().take_front(numModuli));
  return polynomial::RingAttr::get(newRnsType, ringAttr.getPolynomialModulus());
}

polynomial::RingAttr getRlweRNSRingWithLevel(polynomial::RingAttr ringAttr,
                                             int level) {
  return getRlweRNSRingWithLevel(ringAttr, {}, level);
}

FailureOr<polynomial::RingAttr> getRlweRNSRing(MLIRContext* ctx,
                                               ArrayRef<int64_t> primes,
                                               int minSlotCount) {
  // monomial
  std::vector<polynomial::IntMonomial> monomials;
  monomials.emplace_back(1, minSlotCount);
  monomials.emplace_back(1, 0);
  auto result = polynomial::IntPolynomial::fromMonomials(monomials);
  if (failed(result)) return failure();
  polynomial::IntPolynomial xnPlusOne = result.value();

  // moduli chain
  int64_t maxBitWidth = 0;
  for (int64_t prime : primes) {
    maxBitWidth = std::max(maxBitWidth,
                           static_cast<int64_t>(std::ceil(std::log2(prime))));
  }
  // ModArithType requires the underlying integer storage type to have at least
  // 1 extra bit beyond the modulus bit width, and RNSType requires all basis
  // limbs to share the same storage type. Round up to 32 or 64 bits.
  unsigned storageBitWidth =
      std::max<unsigned>(32, llvm::PowerOf2Ceil(maxBitWidth + 1));
  SmallVector<Type, 4> modTypes;
  for (int64_t prime : primes) {
    auto type = IntegerType::get(ctx, storageBitWidth);
    modTypes.push_back(
        mod_arith::ModArithType::get(ctx, IntegerAttr::get(type, prime)));
  }

  // types
  auto rnsType = rns::RNSType::get(ctx, modTypes);
  return polynomial::RingAttr::get(
      rnsType, polynomial::IntPolynomialAttr::get(ctx, xnPlusOne));
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
