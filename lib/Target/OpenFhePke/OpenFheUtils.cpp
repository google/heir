#include "include/Target/OpenFhePke/OpenFheUtils.h"

#include "include/Dialect/LWE/IR/LWETypes.h"
#include "include/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

FailureOr<std::string> convertType(Type type) {
  return llvm::TypeSwitch<Type &, FailureOr<std::string>>(type)
      // For now, these types are defined in the prelude as aliases.
      .Case<CryptoContextType>(
          [&](auto ty) { return std::string("CryptoContextT"); })
      .Case<lwe::RLWECiphertextType>(
          [&](auto ty) { return std::string("CiphertextT"); })
      .Default([&](Type &) { return failure(); });
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
