#include "lib/Dialect/LWE/IR/LWETypes.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

#define DEBUG_TYPE "lwe-types"

namespace mlir {
namespace heir {
namespace lwe {

LogicalResult LWECiphertextType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    Attribute encoding, CiphertextSpaceAttr ciphertextSpace, KeyAttr keyAttr,
    ModulusChainAttr modulusChain) {
  if (!encoding ||
      !mlir::isa<ConstantCoefficientEncodingAttr, CoefficientEncodingAttr,
                 InverseCanonicalEncodingAttr, FullCRTPackingEncodingAttr>(
          encoding)) {
    return emitError() << "expected a valid LWE encoding attribute, but got "
                       << encoding;
  }
  if (keyAttr.getSlotIndex() != 0 && (ciphertextSpace.getSize() != 2)) {
    return emitError() << "a ciphertext with nontrivial slot rotation must "
                          "have size 2, but found size "
                       << ciphertextSpace.getSize();
  }
  if (auto rnsType = mlir::dyn_cast<rns::RNSType>(
          ciphertextSpace.getRing().getCoefficientType())) {
    if (modulusChain &&
        (modulusChain.getCurrent() < 0 || rnsType.getBasisTypes().empty() ||
         rnsType.getBasisTypes().size() - 1 !=
             static_cast<size_t>(modulusChain.getCurrent()))) {
      return emitError()
             << "the level in the ciphertext ring "
                "must match the modulus chain's current, but found rns="
             << rnsType << " and modulus chain=" << modulusChain;
    }
  }
  return success();
}

LWECiphertextType getDefaultCGGICiphertextType(MLIRContext* ctx,
                                               int plaintextBits) {
  auto ciphertextBits = 32;
  auto scalingFactor = 1 << (ciphertextBits - plaintextBits);

  polynomial::IntPolynomial polyX =
      polynomial::IntPolynomial::fromCoefficients({0, 1});
  auto polyXAttr = polynomial::IntPolynomialAttr::get(ctx, polyX);
  auto ciphertextRing = polynomial::RingAttr::get(
      IntegerType::get(ctx, ciphertextBits), polyXAttr);

  return lwe::LWECiphertextType::get(
      ctx,
      lwe::ConstantCoefficientEncodingAttr::get(
          ctx, scalingFactor, static_cast<unsigned>(plaintextBits)),
      lwe::CiphertextSpaceAttr::get(ctx, ciphertextRing,
                                    lwe::LweEncryptionType::msb,
                                    /*dimension=*/742),
      lwe::KeyAttr::get(ctx, 0), /*modulusChain=*/nullptr);
}

std::optional<unsigned> getCGGIPlaintextBitWidth(LWECiphertextType ctType) {
  auto encoding =
      llvm::dyn_cast<ConstantCoefficientEncodingAttr>(ctType.getEncoding());
  if (!encoding || encoding.getPlaintextBits() == 0) {
    return std::nullopt;
  }
  return encoding.getPlaintextBits();
}

LWEPlaintextType getCGGIPlaintextType(LWECiphertextType ctType) {
  auto encoding =
      llvm::cast<ConstantCoefficientEncodingAttr>(ctType.getEncoding());
  unsigned bits = encoding.getPlaintextBits();
  assert(bits != 0 && "CGGI ciphertext must have plaintext_bits set");
  MLIRContext* ctx = ctType.getContext();
  auto polyMod = ctType.getCiphertextSpace().getRing().getPolynomialModulus();
  auto plaintextRing =
      polynomial::RingAttr::get(IntegerType::get(ctx, bits), polyMod);
  auto ptEncoding =
      ConstantCoefficientEncodingAttr::get(ctx, encoding.getScalingFactor());
  return LWEPlaintextType::get(
      ctx, PlaintextSpaceAttr::get(ctx, plaintextRing, ptEncoding));
}

FailureOr<LWECiphertextType> applyModReduce(LWECiphertextType inputType) {
  auto* ctx = inputType.getContext();
  int currentLevel = inputType.getModulusChain().getCurrent();
  int newLevel = inputType.getModulusChain().getCurrent() - 1;
  LLVM_DEBUG(llvm::dbgs() << "Applying mod reduce from level " << currentLevel
                          << " to " << newLevel << "\n");
  if (newLevel < 0) {
    return failure();
  }
  auto ring = inputType.getCiphertextSpace().getRing();
  auto newRing = getRlweRNSRingWithLevel(ring, newLevel);
  LLVM_DEBUG(llvm::dbgs() << "New ring is " << newRing << "\n");

  APInt dividedModulus =
      inputType.getModulusChain().getElements()[currentLevel].getValue();
  lwe::ModulusChainAttr moddedDownChain = lwe::ModulusChainAttr::get(
      ctx, inputType.getModulusChain().getElements(), newLevel);
  LLVM_DEBUG(llvm::dbgs() << "Modded down chain=" << moddedDownChain << "\n");
  Attribute newEncoding = inferModulusSwitchOrRescaleOpEncoding(
      inputType.getEncoding(), dividedModulus);

  LLVM_DEBUG(llvm::dbgs() << "new encoding=" << newEncoding << "\n");
  return lwe::LWECiphertextType::get(
      ctx, newEncoding,
      lwe::CiphertextSpaceAttr::get(
          ctx, newRing, inputType.getCiphertextSpace().getEncryptionType(),
          inputType.getCiphertextSpace().getSize()),
      lwe::KeyAttr::get(ctx, 0),
      lwe::ModulusChainAttr::get(ctx, moddedDownChain.getElements(), newLevel));
}

LWECiphertextType cloneAtLevel(LWECiphertextType inputType, int64_t level) {
  auto* ctx = inputType.getContext();
  auto ring = inputType.getCiphertextSpace().getRing();
  lwe::ModulusChainAttr newChain = lwe::ModulusChainAttr::get(
      ctx, inputType.getModulusChain().getElements(), level);
  auto newRing = getRingFromModulusChain(newChain, ring.getPolynomialModulus());
  return lwe::LWECiphertextType::get(
      ctx, inputType.getEncoding(),
      lwe::CiphertextSpaceAttr::get(
          ctx, newRing, inputType.getCiphertextSpace().getEncryptionType(),
          inputType.getCiphertextSpace().getSize()),
      lwe::KeyAttr::get(ctx, 0), newChain);
}

std::optional<int64_t> getLevel(LWECiphertextType ctType) {
  auto modulusChain = ctType.getModulusChain();
  if (!modulusChain) return std::nullopt;
  return modulusChain.getCurrent();
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
