#include "lib/Conversion/RlweToOpenfhe/RlweToOpenfhe.h"

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

ToOpenfheTypeConverter::ToOpenfheTypeConverter(MLIRContext *ctx) {
  addConversion([](Type type) { return type; });
  addConversion([ctx](lwe::RLWEPublicKeyType type) -> Type {
    return openfhe::PublicKeyType::get(ctx);
  });
  addConversion([ctx](lwe::RLWESecretKeyType type) -> Type {
    return openfhe::PrivateKeyType::get(ctx);
  });
}

FailureOr<Value> getContextualCryptoContext(Operation *op) {
  auto result = getContextualArgFromFunc<openfhe::CryptoContextType>(op);
  if (failed(result)) {
    return op->emitOpError()
           << "Found RLWE op in a function without a public "
              "key argument. Did the AddCryptoContextArg pattern fail to run?";
  }
  return result.value();
}

}  // namespace mlir::heir
