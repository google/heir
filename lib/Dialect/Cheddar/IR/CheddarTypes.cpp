#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"

#include "llvm/include/llvm/ADT/StringRef.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

llvm::StringRef getSupportKind(Type type) {
  return llvm::TypeSwitch<Type, llvm::StringRef>(type)
      .Case<ContextType>([](auto) { return ContextType::getMnemonic(); })
      .Case<BootContextType>(
          [](auto) { return BootContextType::getMnemonic(); })
      .Case<EncoderType>([](auto) { return EncoderType::getMnemonic(); })
      .Case<UserInterfaceType>(
          [](auto) { return UserInterfaceType::getMnemonic(); })
      .Case<EvalKeyType>([](auto) { return EvalKeyType::getMnemonic(); })
      .Case<EvkMapType>([](auto) { return EvkMapType::getMnemonic(); })
      .Default([](Type) { return llvm::StringRef(); });
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
