#include "lib/Target/OpenFhePke/OpenFheUtils.h"

#include <string>

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

std::string getModulePrelude(OpenfheScheme scheme) {
  return llvm::formatv(kModulePreludeTemplate.data(),
                       scheme == OpenfheScheme::CKKS ? "CKKS" : "BGV");
}

FailureOr<std::string> convertType(Type type) {
  return llvm::TypeSwitch<Type &, FailureOr<std::string>>(type)
      // For now, these types are defined in the prelude as aliases.
      .Case<CryptoContextType>(
          [&](auto ty) { return std::string("CryptoContextT"); })
      .Case<CCParamsType>([&](auto ty) { return std::string("CCParamsT"); })
      .Case<lwe::RLWECiphertextType>(
          [&](auto ty) { return std::string("CiphertextT"); })
      .Case<lwe::RLWEPlaintextType>(
          [&](auto ty) { return std::string("Plaintext"); })
      .Case<openfhe::EvalKeyType>(
          [&](auto ty) { return std::string("EvalKeyT"); })
      .Case<openfhe::PrivateKeyType>(
          [&](auto ty) { return std::string("PrivateKeyT"); })
      .Case<openfhe::PublicKeyType>(
          [&](auto ty) { return std::string("PublicKeyT"); })
      .Case<IndexType>([&](auto ty) { return std::string("size_t"); })
      .Case<IntegerType>([&](auto ty) {
        auto width = ty.getWidth();
        if (width != 8 && width != 16 && width != 32 && width != 64) {
          return FailureOr<std::string>();
        }
        SmallString<8> result;
        llvm::raw_svector_ostream os(result);
        os << "int" << width << "_t";
        return FailureOr<std::string>(std::string(result));
      })
      .Case<FloatType>([&](auto ty) -> FailureOr<std::string> {
        auto width = ty.getWidth();
        if (width == 32) {
          return std::string("float");
        } else if (width == 64) {
          return std::string("double");
        }
        return failure();
      })
      .Case<RankedTensorType>([&](auto ty) {
        auto eltTyResult = convertType(ty.getElementType());
        if (failed(eltTyResult)) {
          return FailureOr<std::string>();
        }
        auto result = eltTyResult.value();
        for (int i = 0; i < ty.getRank(); ++i) {
          // For each dimension, wrap the element in a std::vector.
          result = "std::vector<" + result + ">";
        }
        return FailureOr<std::string>(std::string(result));
      })
      .Default([&](Type &) { return failure(); });
}

FailureOr<Value> getContextualCryptoContext(Operation *op) {
  Value cryptoContext = op->getParentOfType<func::FuncOp>()
                            .getBody()
                            .getBlocks()
                            .front()
                            .getArguments()
                            .front();
  if (!mlir::isa<openfhe::CryptoContextType>(cryptoContext.getType())) {
    return op->emitOpError()
           << "Found op in a function without a crypto context argument.";
  }
  return cryptoContext;
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
