#include "lib/Target/Lattigo/LattigoDebugEmitter.h"

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ios>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/Lattigo/LattigoTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/CommandLine.h"     // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"   // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/ManagedStatic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

LogicalResult translateToDebugEmitter(mlir::Operation* op,
                                      llvm::raw_ostream& os,
                                      const std::string& packageName) {
  LattigoDebugEmitter emitter(os, packageName);
  LogicalResult result = emitter.translate(*op);
  return result;
}

FailureOr<std::string> LattigoDebugEmitter::convertType(Type type) {
  return llvm::TypeSwitch<Type, FailureOr<std::string>>(type)
      // RLWE
      .Case<RLWECiphertextType>(
          [&](auto ty) { return std::string("*rlwe.Ciphertext"); })
      .Case<RLWEPlaintextType>(
          [&](auto ty) { return std::string("*rlwe.Plaintext"); })
      .Case<RLWESecretKeyType>(
          [&](auto ty) { return std::string("*rlwe.PrivateKey"); })
      .Case<RLWEPublicKeyType>(
          [&](auto ty) { return std::string("*rlwe.PublicKey"); })
      .Case<RLWEKeyGeneratorType>(
          [&](auto ty) { return std::string("*rlwe.KeyGenerator"); })
      .Case<RLWERelinearizationKeyType>(
          [&](auto ty) { return std::string("*rlwe.RelinearizationKey"); })
      .Case<RLWEGaloisKeyType>(
          [&](auto ty) { return std::string("*rlwe.GaloisKey"); })
      .Case<RLWEEvaluationKeySetType>(
          [&](auto ty) { return std::string("*rlwe.EvaluationKeySet"); })
      .Case<RLWEEncryptorType>(
          [&](auto ty) { return std::string("*rlwe.Encryptor"); })
      .Case<RLWEDecryptorType>(
          [&](auto ty) { return std::string("*rlwe.Decryptor"); })
      .Case<BGVEncoderType>(
          [&](auto ty) { return std::string("*bgv.Encoder"); })
      .Case<BGVEvaluatorType>(
          [&](auto ty) { return std::string("*bgv.Evaluator"); })
      .Case<BGVParameterType>(
          [&](auto ty) { return std::string("bgv.Parameters"); })
      .Case<CKKSEncoderType>(
          [&](auto ty) { return std::string("*ckks.Encoder"); })
      .Case<CKKSEvaluatorType>(
          [&](auto ty) { return std::string("*ckks.Evaluator"); })
      .Case<CKKSBootstrappingEvaluationKeysType>(
          [&](auto ty) { return std::string("*bootstrapping.EvaluationKeys"); })
      .Case<CKKSBootstrappingEvaluatorType>(
          [&](auto ty) { return std::string("*bootstrapping.Evaluator"); })
      .Case<CKKSParameterType>(
          [&](auto ty) { return std::string("ckks.Parameters"); })
      .Case<CKKSBootstrappingParameterType>(
          [&](auto ty) { return std::string("bootstrapping.Parameters"); })
      .Default([&](Type) -> FailureOr<std::string> { return failure(); });
}

LogicalResult LattigoDebugEmitter::emitDebugHelperSignature(
    ::mlir::func::FuncOp funcOp, ErrorEmitterFn emitError) {
  auto argTypes = funcOp.getArgumentTypes();

  if (argTypes.size() != 5) {
    return emitError(
        funcOp.getLoc(),
        llvm::formatv(
            "Unexpected debug port signature: expected 5 args, got {0}",
            argTypes.size()));
  }

  llvm::SmallVector<std::string, 5> funcArgs;
  for (size_t i = 0; i < argTypes.size(); i++) {
    auto param = convertType(argTypes[i]);
    if (failed(param))
      return emitError(
          funcOp.getLoc(),
          llvm::formatv("Failed to emit type for arg{0}: {1}", i, argTypes[i]));

    funcArgs.push_back(param.value());
  }

  os << "func";
  os << " " << canonicalizeDebugPort(funcOp.getName()) << "(";

  os << kEvalVar << " " << funcArgs[0] << ", ";
  os << kParamVar << " " << funcArgs[1] << ", ";
  os << kEncodeVar << " " << funcArgs[2] << ", ";
  os << kDecryptVar << " " << funcArgs[3] << ", ";
  os << kCiphertxtVar << " " << funcArgs[4] << ", ";
  os << kDebugAttrMapParam;
  os << " " << "map[string]string";
  os << ")";
  return success();
}

LogicalResult LattigoDebugEmitter::emitDebugHelperImpl() {
  os << "isBlockArgument" << " := " << kDebugAttrMapParam
     << "[\"asm.is_block_arg\"]\n";

  os << "if isBlockArgument == \"1\" {\n";
  os.indent();
  os << "fmt.Println(\"Input\")\n";
  os.unindent();
  os << "} else {\n";
  os.indent();
  os << "fmt.Println(" << kDebugAttrMapParam << "[\"asm.op_name\"])\n";
  os.unindent();
  os << "}\n\n";

  os << "messageSize, _ := strconv.Atoi(" << kDebugAttrMapParam
     << "[\"message.size\"])\n";
  os << "value := make([]int64, messageSize)\n";
  os << "pt := " << kDecryptVar << ".DecryptNew(" << kCiphertxtVar << ")\n";
  os << kEncodeVar << ".Decode(pt, value)\n";
  os << "fmt.Printf(\"  %v\\n\", value)\n";
  return success();
}

LogicalResult LattigoDebugEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp>([&](auto op) { return printOperation(op); })
          .Default([&](Operation&) {
            return emitError(op.getLoc(), "unable to find printer for op");
          });

  if (failed(status)) {
    return emitError(op.getLoc(),
                     llvm::formatv("Failed to translate op {0}", op.getName()));
  }
  return success();
}

LogicalResult LattigoDebugEmitter::printOperation(ModuleOp moduleOp) {
  prelude = "package " + packageName + "\n";
  imports.insert("\"fmt\"");
  imports.insert("\"strconv\"");

  imports.insert(std::string(kRlweImport));
  if (moduleIsBGVOrBFV(moduleOp)) {
    imports.insert(std::string(kBgvImport));
  } else if (moduleIsCKKS(moduleOp)) {
    imports.insert(std::string(kCkksImport));
  } else {
    return moduleOp.emitError("Unknown scheme");
  }

  emitPrelude();

  for (Operation& op : moduleOp) {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }

  return success();
}

LogicalResult LattigoDebugEmitter::printOperation(func::FuncOp funcOp) {
  if (!isDebugPort(funcOp.getName()) || isEmitted) {
    return success();
  }

  auto res = emitDebugHelperSignature(
      funcOp, [&](Location loc, const std::string& message) {
        return emitError(loc, message);
      });

  if (failed(res)) {
    return res;
  }

  os << " {\n";
  os.indent();
  res = emitDebugHelperImpl();
  if (failed(res)) {
    return res;
  }
  os.unindent();
  os << "}\n";
  isEmitted = true;
  return success();
}

LattigoDebugEmitter::LattigoDebugEmitter(raw_ostream& os,
                                         const std::string& packageName)
    : os(os), packageName(packageName), isEmitted(false) {}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
