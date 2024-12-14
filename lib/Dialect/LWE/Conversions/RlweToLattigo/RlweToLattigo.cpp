#include "lib/Dialect/LWE/Conversions/RlweToLattigo/RlweToLattigo.h"

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

ToLattigoTypeConverter::ToLattigoTypeConverter(MLIRContext *ctx) {
  addConversion([](Type type) { return type; });
  addConversion([ctx](lwe::NewLWECiphertextType type) -> Type {
    return lattigo::RLWECiphertextType::get(ctx);
  });
  addConversion([ctx](lwe::NewLWEPlaintextType type) -> Type {
    return lattigo::RLWEPlaintextType::get(ctx);
  });
  addConversion([ctx](lwe::NewLWEPublicKeyType type) -> Type {
    return lattigo::RLWEPublicKeyType::get(ctx);
  });
  addConversion([ctx](lwe::NewLWESecretKeyType type) -> Type {
    return lattigo::RLWESecretKeyType::get(ctx);
  });
}

}  // namespace mlir::heir
