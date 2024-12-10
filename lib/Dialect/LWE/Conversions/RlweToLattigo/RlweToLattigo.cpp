#include "lib/Dialect/LWE/Conversions/RlweToLattigo/RlweToLattigo.h"

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

ToLattigoTypeConverter::ToLattigoTypeConverter(MLIRContext *ctx) {
  addConversion([](Type type) { return type; });
  addConversion([ctx](lwe::RLWEPublicKeyType type) -> Type {
    return lattigo::RLWEPublicKeyType::get(ctx);
  });
  addConversion([ctx](lwe::RLWESecretKeyType type) -> Type {
    return lattigo::RLWESecretKeyType::get(ctx);
  });
  addConversion([ctx](lwe::RLWECiphertextType type) -> Type {
    return lattigo::RLWECiphertextType::get(ctx);
  });
  addConversion([ctx](lwe::RLWEPlaintextType type) -> Type {
    return lattigo::RLWEPlaintextType::get(ctx);
  });
}

}  // namespace mlir::heir
