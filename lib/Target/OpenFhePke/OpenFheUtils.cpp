#include "lib/Target/OpenFhePke/OpenFheUtils.h"

#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "lib/Utils/TargetUtils.h"
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
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
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
  return llvm::TypeSwitch<Type&, FailureOr<std::string>>(type)
      // For now, these types are defined in the prelude as aliases.
      .Case<CryptoContextType>(
          [&](auto ty) { return std::string("CryptoContextT"); })
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
        // std::allocator does not support const types
        auto eltTyResult = convertType(ty.getElementType(), loc, false);
        if (failed(eltTyResult)) {
          return FailureOr<std::string>();
        }
        auto result = "std::vector<" + eltTyResult.value() + ">";
        return FailureOr<std::string>(std::string(result));
      })
      .Default([&](Type&) { return failure(); });
}

FailureOr<Value> getContextualCryptoContext(Operation* op) {
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

LogicalResult funcDeclarationHelper(::mlir::func::FuncOp funcOp,
                                    ::mlir::raw_indented_ostream& os,
                                    SelectVariableNames* variableNames,
                                    TypeEmitterFn emitType,
                                    ErrorEmitterFn emitError) {
  if (funcOp.getNumResults() > 1) {
    return emitError(funcOp.getLoc(),
                     llvm::formatv("Only functions with <= 1 return type "
                                   "are supported, but this function has ",
                                   funcOp.getNumResults()));
    return failure();
  }

  if (funcOp.getNumResults() == 1) {
    Type result = funcOp.getResultTypes()[0];
    if (failed(emitType(result, funcOp->getLoc()))) {
      return emitError(funcOp.getLoc(),
                       llvm::formatv("Failed to emit type {0}", result));
    }
  } else {
    os << "void";
  }

  os << " " << canonicalizeDebugPort(funcOp.getName()) << "(";

  // Check the types without printing to enable failure outside of
  // commaSeparatedValues; maybe consider making commaSeparatedValues combine
  // the results into a FailureOr, like commaSeparatedTypes in tfhe_rust
  // emitter.
  for (Value arg : funcOp.getArguments()) {
    if (failed(convertType(arg.getType(), arg.getLoc()))) {
      return emitError(funcOp.getLoc(),
                       llvm::formatv("Failed to emit type {0}", arg.getType()));
    }
  }

  if (funcOp.isDeclaration()) {
    // function declaration
    os << commaSeparatedTypes(funcOp.getArgumentTypes(), [&](Type type) {
      return convertType(type, funcOp->getLoc()).value();
    });
    // debug attribute map for debug call
    if (isDebugPort(funcOp.getName())) {
      os << ", const std::map<std::string, std::string>&";
    }
  } else {
    os << commaSeparatedValues(funcOp.getArguments(), [&](Value value) {
      return convertType(value.getType(), funcOp->getLoc()).value() + " " +
             variableNames->getNameForValue(value);
    });
  }
  os << ")";
  return success();
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
