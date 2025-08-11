#include "lib/Target/OpenFhePke/OpenFheUtils.h"

#include <string>

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

std::string getModulePrelude(OpenfheScheme scheme,
                             OpenfheImportType importType) {
  auto import = importType == OpenfheImportType::SOURCE_RELATIVE
                    ? kSourceRelativeOpenfheImport
                    : (importType == OpenfheImportType::INSTALL_RELATIVE
                           ? kInstallationRelativeOpenfheImport
                           : kEmbeddedOpenfheImport);
  auto prelude = std::string(
      llvm::formatv(kModulePreludeTemplate.data(),
                    scheme == OpenfheScheme::CKKS
                        ? "CKKS"
                        : (scheme == OpenfheScheme::BGV ? "BGV" : "BFV")));
  return std::string(import) + prelude;
}

std::string getWeightsPrelude() { return std::string(kWeightsPreludeTemplate); }

FailureOr<std::string> convertType(Type type, Location loc, bool constant) {
  return llvm::TypeSwitch<Type &, FailureOr<std::string>>(type)
      // For now, these types are defined in the prelude as aliases.
      .Case<CryptoContextType>(
          [&](auto ty) { return std::string("CryptoContextT"); })
      .Case<BinFHEContextType>(
          [&](auto ty) { return std::string("BinFHEContextT"); })
      .Case<LWESchemeType>([&](auto ty) { return std::string("LWESchemeT"); })
      .Case<LookupTableType>(
          [&](auto ty) { return std::string("NativeVector"); })
      .Case<CCParamsType>([&](auto ty) { return std::string("CCParamsT"); })
      .Case<lwe::LWECiphertextType>([&](auto ty) {
        return constant ? std::string("CiphertextT")
                        : std::string("MutableCiphertextT");
      })
      .Case<lwe::LWEPlaintextType>(
          [&](auto ty) { return std::string("Plaintext"); })
      .Case<openfhe::EvalKeyType>(
          [&](auto ty) { return std::string("EvalKeyT"); })
      .Case<openfhe::PrivateKeyType>(
          [&](auto ty) { return std::string("PrivateKeyT"); })
      .Case<openfhe::PublicKeyType>(
          [&](auto ty) { return std::string("PublicKeyT"); })
      .Case<IndexType>([&](auto ty) { return std::string("size_t"); })
      .Case<IntegerType>([&](auto ty) -> FailureOr<std::string> {
        auto width = ty.getWidth();
        if (width == 1) {
          return std::string("bool");
        }
        if (width != 8 && width != 16 && width != 32 && width != 64) {
          return failure();
        }
        SmallString<8> result;
        llvm::raw_svector_ostream os(result);
        os << "int" << width << "_t";
        return std::string(result);
      })
      .Case<FloatType>([&](auto ty) -> FailureOr<std::string> {
        auto width = ty.getWidth();
        switch (width) {
          case 8:
          case 16:
            emitWarning(
                loc,
                "Floating point width " + std::to_string(width) +
                    " is not supported in C++, using 32-bit float instead.");
            [[fallthrough]];
          case 32:
            return std::string("float");
          case 64:
            return std::string("double");
          default:
            return failure();
        }
      })
      .Case<RankedTensorType>([&](auto ty) {
        auto eltTyResult = convertType(ty.getElementType(), loc);
        if (failed(eltTyResult)) {
          return FailureOr<std::string>();
        }
        auto result = "std::vector<" + eltTyResult.value() + ">";
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
