#include "lib/Target/OpenFhePke/OpenFheBinHeaderEmitter.h"

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"  // from @heir
#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

std::string prelude = R"cpp(
#include "src/binfhe/include/binfhecontext.h"   // from @openfhe
#include "src/binfhe/include/lwe-privatekey.h"  // from @openfhe
#include "src/binfhe/include/lwe-publickey.h"   // from @openfhe
#include "src/binfhe/include/lwe-ciphertext.h"  // from @openfhe

  using namespace lbcrypto;

  using BinFHEContextT = std::shared_ptr<BinFHEContext>;
  using LWESchemeT = std::shared_ptr<LWEEncryptionScheme>;
  using CiphertextT = LWECiphertext;

  std::vector<LWECiphertext> encrypt(BinFHEContextT cc, LWEPrivateKey sk,
                                     int value, int width = 8);
  int decrypt(BinFHEContextT cc, LWEPrivateKey sk,
              std::vector<LWECiphertext> encrypted);
)cpp";

// Registration is done in OpenFheTranslateRegistration.cpp

LogicalResult translateToOpenFheBinHeader(Operation *op,
                                          llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  OpenFheBinHeaderEmitter emitter(os, &variableNames);
  return emitter.translate(*op);
}

LogicalResult OpenFheBinHeaderEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          .Case<func::FuncOp>([&](auto op) { return printOperation(op); })
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult OpenFheBinHeaderEmitter::printOperation(ModuleOp moduleOp) {
  os << prelude << "\n";
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult OpenFheBinHeaderEmitter::printOperation(func::FuncOp funcOp) {
  // If keeping this consistent alongside OpenFheEmitter gets annoying,
  // extract to a shared function in a base class.
  if (funcOp.getNumResults() != 1) {
    return funcOp.emitOpError() << "Only functions with a single return type "
                                   "are supported, but this function has "
                                << funcOp.getNumResults();
    return failure();
  }

  Type result = funcOp.getResultTypes()[0];
  if (failed(emitType(result, funcOp->getLoc()))) {
    return funcOp.emitOpError() << "Failed to emit type " << result;
  }

  os << " " << funcOp.getName() << "(";

  for (Value arg : funcOp.getArguments()) {
    if (failed(convertType(arg.getType(), funcOp.getLoc()))) {
      return funcOp.emitOpError() << "Failed to emit type " << arg.getType();
    }
  }

  os << commaSeparatedValues(funcOp.getArguments(), [&](Value value) {
    auto res = convertType(value.getType(), funcOp.getLoc());
    return res.value() + " " + variableNames->getNameForValue(value);
  });
  os << ");\n";

  return success();
}

LogicalResult OpenFheBinHeaderEmitter::emitType(Type type, Location loc) {
  auto result = convertType(type, loc);
  if (failed(result)) {
    return failure();
  }
  // convertType returns FailureOr<std::string>; emit the contained value
  os << result.value();
  return success();
}

OpenFheBinHeaderEmitter::OpenFheBinHeaderEmitter(
    raw_ostream &os, SelectVariableNames *variableNames)
    : os(os), variableNames(variableNames) {}
}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
