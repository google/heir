#include "lib/Dialect/LWE/IR/LWETypes.h"

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

LogicalResult LWECiphertextType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::heir::lwe::ApplicationDataAttr, mlir::heir::lwe::PlaintextSpaceAttr,
    mlir::heir::lwe::CiphertextSpaceAttr ciphertextSpace,
    mlir::heir::lwe::KeyAttr keyAttr,
    mlir::heir::lwe::ModulusChainAttr modulusChain) {
  if (keyAttr.getSlotIndex() != 0 && (ciphertextSpace.getSize() != 2)) {
    return emitError() << "a ciphertext with nontrivial slot rotation must "
                          "have size 2, but found size "
                       << ciphertextSpace.getSize();
  }
  if (auto rnsType = mlir::dyn_cast<rns::RNSType>(
          ciphertextSpace.getRing().getCoefficientType())) {
    if (rnsType.getBasisTypes().size() - 1 != modulusChain.getCurrent()) {
      return emitError() << "the level in the ciphertext ring "
                            "must match the modulus chain's current, but found "
                         << rnsType.getBasisTypes().size() - 1 << " and "
                         << modulusChain.getCurrent();
    }
  }
  return success();
}

LWECiphertextType getDefaultCGGICiphertextType(MLIRContext *ctx,
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

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
