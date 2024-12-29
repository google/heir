#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"

#include "lib/Dialect/Openfhe/IR/OpenfheDialect.cpp.inc"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/Openfhe/IR/OpenfheOps.cpp.inc"

namespace mlir {
namespace heir {
namespace openfhe {

std::string openfheSuggestNameForType(Type type) {
  return llvm::TypeSwitch<Type, std::string>(type)
      .Case<CryptoContextType>([&](Type) { return "cc"; })
      .Case<CCParamsType>([&](Type) { return "params"; })
      .Case<PublicKeyType>([&](Type) { return "pk"; })
      .Case<PrivateKeyType>([&](Type) { return "sk"; })
      .Default([&](Type) { return ""; });  // use the default numbering.
}

void OpenfheDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Openfhe/IR/OpenfheOps.cpp.inc"
      >();
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
