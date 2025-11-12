#include "lib/Dialect/LWE/IR/LWETypes.h"

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

#define DEBUG_TYPE "lwe-types"

namespace mlir {
namespace heir {
namespace lwe {

LogicalResult LWECiphertextType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ApplicationDataAttr, PlaintextSpaceAttr,
    CiphertextSpaceAttr ciphertextSpace, KeyAttr keyAttr,
    ModulusChainAttr modulusChain) {
  if (keyAttr.getSlotIndex() != 0 && (ciphertextSpace.getSize() != 2)) {
    return emitError() << "a ciphertext with nontrivial slot rotation must "
                          "have size 2, but found size "
                       << ciphertextSpace.getSize();
  }
  if (auto rnsType = mlir::dyn_cast<rns::RNSType>(
          ciphertextSpace.getRing().getCoefficientType())) {
    if (rnsType.getBasisTypes().size() - 1 != modulusChain.getCurrent()) {
      return emitError()
             << "the level in the ciphertext ring "
                "must match the modulus chain's current, but found rns="
             << rnsType << " and modulus chain=" << modulusChain;
    }
  }
  return success();
}

LWECiphertextType getDefaultCGGICiphertextType(MLIRContext* ctx,
                                               int messageWidth,
                                               int plaintextBits) {
  auto ciphertextBits = 32;
  auto scalingFactor = 1 << (ciphertextBits - plaintextBits);

  polynomial::IntPolynomial polyX =
      polynomial::IntPolynomial::fromCoefficients({1, 0});
  auto polyXAttr = polynomial::IntPolynomialAttr::get(ctx, polyX);
  auto plaintextRing = polynomial::RingAttr::get(
      IntegerType::get(ctx, plaintextBits), polyXAttr);
  auto ciphertextRing = polynomial::RingAttr::get(
      IntegerType::get(ctx, ciphertextBits), polyXAttr);

  return lwe::LWECiphertextType::get(
      ctx,
      lwe::ApplicationDataAttr::get(ctx, IntegerType::get(ctx, messageWidth),
                                    lwe::PreserveOverflowAttr::get(ctx)),
      lwe::PlaintextSpaceAttr::get(
          ctx, plaintextRing,
          lwe::ConstantCoefficientEncodingAttr::get(ctx, scalingFactor)),
      lwe::CiphertextSpaceAttr::get(ctx, ciphertextRing,
                                    lwe::LweEncryptionType::msb,
                                    /*dimension=*/742),
      lwe::KeyAttr::get(ctx, 0), /*modulusChain=*/nullptr);
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
  lwe::PlaintextSpaceAttr newPlaintextSpace =
      inferModulusSwitchOrRescaleOpPlaintextSpaceAttr(
          ctx, inputType.getPlaintextSpace(), dividedModulus);

  LLVM_DEBUG(llvm::dbgs() << "new plaintext space=" << newPlaintextSpace
                          << "\n");
  return lwe::LWECiphertextType::get(
      ctx, inputType.getApplicationData(), newPlaintextSpace,
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
      ctx, inputType.getApplicationData(), inputType.getPlaintextSpace(),
      lwe::CiphertextSpaceAttr::get(
          ctx, newRing, inputType.getCiphertextSpace().getEncryptionType(),
          inputType.getCiphertextSpace().getSize()),
      lwe::KeyAttr::get(ctx, 0), newChain);
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
