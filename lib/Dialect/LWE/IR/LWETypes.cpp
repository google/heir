#include "lib/Dialect/LWE/IR/LWETypes.h"

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

LogicalResult NewLWECiphertextType::verify(
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

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
