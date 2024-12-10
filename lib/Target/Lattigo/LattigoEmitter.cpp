#include "lib/Target/Lattigo/LattigoEmitter.h"

#include <string>
#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Target/Lattigo/LattigoTemplates.h"
#include "lib/Utils/TargetUtils/TargetUtils.h"
#include "llvm/include/llvm/ADT/StringExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
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
namespace lattigo {

LogicalResult translateToLattigo(Operation *op, llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  LattigoEmitter emitter(os, &variableNames);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult LattigoEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp, func::CallOp>(
              [&](auto op) { return printOperation(op); })
          .Case<arith::ConstantOp>([&](auto op) { return printOperation(op); })
          // Lattigo ops
          .Case<RLWENewEncryptorOp, RLWENewDecryptorOp, RLWENewKeyGeneratorOp,
                RLWEGenKeyPairOp, RLWEEncryptOp, RLWEDecryptOp,
                BGVNewParametersFromLiteralOp, BGVNewEncoderOp,
                BGVNewEvaluatorOp, BGVNewPlaintextOp, BGVEncodeOp, BGVDecodeOp,
                BGVAddOp, BGVSubOp, BGVMulOp>(
              [&](auto op) { return printOperation(op); })
          .Default([&](Operation &) {
            return emitError(op.getLoc(), "unable to find printer for op");
          });

  if (failed(status)) {
    return emitError(op.getLoc(),
                     llvm::formatv("Failed to translate op {0}", op.getName()));
  }
  return success();
}

LogicalResult LattigoEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePreludeTemplate;

  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult LattigoEmitter::printOperation(func::FuncOp funcOp) {
  // name and arg
  os << "func " << funcOp.getName() << "(";
  os << getCommaSeparatedNamesWithTypes(funcOp.getArguments());
  os << ") ";

  // return types
  auto resultTypesString = getCommaSeparatedTypes(funcOp.getResultTypes());
  if (failed(resultTypesString)) {
    return failure();
  }
  if (!resultTypesString->empty()) {
    os << "(";
    os << resultTypesString.value();
    os << ") ";
  }
  os << "{\n";
  os.indent();

  // body
  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }

  os.unindent();
  os << "}\n";

  return success();
}

LogicalResult LattigoEmitter::printOperation(func::ReturnOp op) {
  os << "return ";
  os << getCommaSeparatedNames(op.getOperands());
  os << "\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(func::CallOp op) {
  if (op.getNumResults() > 0) {
    os << getCommaSeparatedNames(op.getResults());
    os << " := ";
  }
  os << op.getCallee() << "(";
  os << getCommaSeparatedNames(op.getOperands());
  os << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  std::string valueString;
  auto res = llvm::TypeSwitch<Attribute, LogicalResult>(valueAttr)
                 .Case<IntegerAttr>([&](IntegerAttr intAttr) {
                   valueString = std::to_string(intAttr.getInt());
                   return success();
                 })
                 .Case<DenseElementsAttr>([&](DenseElementsAttr denseAttr) {
                   valueString = "[]int64{";
                   for (auto value : denseAttr.getValues<APInt>()) {
                     valueString += std::to_string(value.getSExtValue()) + ", ";
                   }
                   // remote the trailing ", "
                   if (valueString.size() > 1) {
                     valueString.pop_back();
                     valueString.pop_back();
                   }
                   valueString += "}";
                   return success();
                 })
                 .Default([&](auto) { return failure(); });
  if (failed(res)) {
    return res;
  }
  os << getName(op.getResult()) << " := " << valueString << "\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(RLWENewEncryptorOp op) {
  return printNewMethod(op.getResult(), {op.getParams(), op.getPublicKey()},
                        "rlwe.NewEncryptor", false);
}

LogicalResult LattigoEmitter::printOperation(RLWENewDecryptorOp op) {
  return printNewMethod(op.getResult(), {op.getParams(), op.getSecretKey()},
                        "rlwe.NewDecryptor", false);
}

LogicalResult LattigoEmitter::printOperation(RLWENewKeyGeneratorOp op) {
  return printNewMethod(op.getResult(), {op.getParams()},
                        "rlwe.NewKeyGenerator", false);
}

LogicalResult LattigoEmitter::printOperation(RLWEGenKeyPairOp op) {
  return printEvalNewMethod(op.getResults(), op.getKeyGenerator(), {},
                            "GenKeyPairNew", false);
}

LogicalResult LattigoEmitter::printOperation(RLWEEncryptOp op) {
  return printEvalNewMethod(op.getResult(), op.getEncryptor(),
                            {op.getPlaintext()}, "EncryptNew", true);
}

LogicalResult LattigoEmitter::printOperation(RLWEDecryptOp op) {
  return printEvalNewMethod(op.getResult(), op.getDecryptor(),
                            {op.getCiphertext()}, "DecryptNew", false);
}

LogicalResult LattigoEmitter::printOperation(BGVNewEncoderOp op) {
  return printNewMethod(op.getResult(), {op.getParams()}, "bgv.NewEncoder",
                        false);
}

LogicalResult LattigoEmitter::printOperation(BGVNewEvaluatorOp op) {
  os << getName(op.getResult()) << " := " << "bgv.NewEvaluator(";
  os << getName(op.getParams()) << ", nil)\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVNewPlaintextOp op) {
  os << getName(op.getResult()) << " := " << "bgv.NewPlaintext(";
  os << getName(op.getParams()) << ", ";
  os << getName(op.getParams()) << ".MaxLevel()";
  os << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVEncodeOp op) {
  return printEvalInplaceMethod(op.getEncoded(), op.getEncoder(), op.getValue(),
                                op.getPlaintext(), "Encode", false);
}

LogicalResult LattigoEmitter::printOperation(BGVDecodeOp op) {
  return printEvalInplaceMethod(op.getDecoded(), op.getEncoder(),
                                op.getPlaintext(), op.getValue(), "Decode",
                                false);
}

LogicalResult LattigoEmitter::printOperation(BGVAddOp op) {
  return printEvalNewMethod(op.getResult(), op.getEvaluator(),
                            {op.getLhs(), op.getRhs()}, "AddNew", true);
}

LogicalResult LattigoEmitter::printOperation(BGVSubOp op) {
  return printEvalNewMethod(op.getResult(), op.getEvaluator(),
                            {op.getLhs(), op.getRhs()}, "SubNew", true);
}

LogicalResult LattigoEmitter::printOperation(BGVMulOp op) {
  return printEvalNewMethod(op.getResult(), op.getEvaluator(),
                            {op.getLhs(), op.getRhs()}, "MulNew", true);
}

std::string printDenseI32ArrayAttr(DenseI32ArrayAttr attr) {
  std::string res = "[]int{";
  res += commaSeparated(attr.asArrayRef());
  res += "}";
  return res;
}

// use i64 for u64 now
std::string printDenseU64ArrayAttr(DenseI64ArrayAttr attr) {
  std::string res = "[]uint64{";
  res += commaSeparated(attr.asArrayRef());
  res += "}";
  return res;
}

LogicalResult LattigoEmitter::printOperation(BGVNewParametersFromLiteralOp op) {
  std::string errResult = ", err";
  os << getName(op.getResult()) << errResult
     << " := bgv.NewParametersFromLiteral(";
  os << "bgv.ParametersLiteral{\n";
  os.indent();
  os << "LogN: " << op.getParamsLiteral().getLogN() << ",\n";
  if (auto Q = op.getParamsLiteral().getQ()) {
    os << "Q: " << printDenseU64ArrayAttr(Q) << ",\n";
  }
  if (auto P = op.getParamsLiteral().getP()) {
    os << "P: " << printDenseU64ArrayAttr(P) << ",\n";
  }
  if (auto LogQ = op.getParamsLiteral().getLogQ()) {
    os << "LogQ: " << printDenseI32ArrayAttr(LogQ) << ",\n";
  }
  if (auto LogP = op.getParamsLiteral().getLogP()) {
    os << "LogP: " << printDenseI32ArrayAttr(LogP) << ",\n";
  }
  os << "PlaintextModulus: " << op.getParamsLiteral().getPlaintextModulus()
     << ",\n";
  os.unindent();
  os << "})\n";
  printErrPanic();
  return success();
}

void LattigoEmitter::printErrPanic() {
  os << "if err != nil {\n";
  os.indent();
  os << "panic(err)\n";
  os.unindent();
  os << "}\n";
}

LogicalResult LattigoEmitter::printNewMethod(::mlir::Value result,
                                             ::mlir::ValueRange operands,
                                             std::string_view op, bool err) {
  std::string errResult = err ? ", err" : "";
  os << getName(result);
  os << errResult << " := " << op << "(";
  os << getCommaSeparatedNames(operands);
  os << ")\n";
  if (err) {
    printErrPanic();
  }
  return success();
}

LogicalResult LattigoEmitter::printEvalInplaceMethod(
    ::mlir::Value result, ::mlir::Value evaluator, ::mlir::Value operand,
    ::mlir::Value operandInplace, std::string_view op, bool err) {
  std::string errResult = err ? ", err := " : "";
  os << errResult << getName(evaluator) << "." << op << "(" << getName(operand)
     << ", " << getName(operandInplace) << ");\n";
  if (err) {
    printErrPanic();
  }
  // for inplace operation, operandInplace is actually a pointer in GO
  // so assigning it is not costly, but for lowering pass to Lattigo, they
  // should ensure the operandInplace is only used once
  os << getName(result) << " := " << getName(operandInplace) << "\n";
  return success();
}

LogicalResult LattigoEmitter::printEvalNewMethod(::mlir::ValueRange results,
                                                 ::mlir::Value evaluator,
                                                 ::mlir::ValueRange operands,
                                                 std::string_view op,
                                                 bool err) {
  std::string errResult = err ? ", err" : "";
  os << getCommaSeparatedNames(results);
  os << errResult << " := " << getName(evaluator) << "." << op << "(";
  os << getCommaSeparatedNames(operands);
  os << ")\n";
  if (err) {
    printErrPanic();
  }
  return success();
}

FailureOr<std::string> LattigoEmitter::convertType(Type type) {
  return llvm::TypeSwitch<Type, FailureOr<std::string>>(type)
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
      .Case<IntegerType>([&](auto ty) -> FailureOr<std::string> {
        auto width = ty.getWidth();
        if (width != 8 && width != 16 && width != 32 && width != 64) {
          return failure();
        }
        return "int" + std::to_string(width);
      })
      .Case<RankedTensorType>([&](auto ty) -> FailureOr<std::string> {
        auto eltTyResult = convertType(ty.getElementType());
        if (failed(eltTyResult)) {
          return failure();
        }
        auto result = eltTyResult.value();
        return std::string("[]") + result;
      })
      .Default([&](Type) -> FailureOr<std::string> { return failure(); });
}

LogicalResult LattigoEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

LattigoEmitter::LattigoEmitter(raw_ostream &os,
                               SelectVariableNames *variableNames)
    : os(os), variableNames(variableNames) {}

void registerToLattigoTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-lattigo",
      "translate the lattigo dialect to GO code against the Lattigo API",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToLattigo(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, func::FuncDialect,
                        tensor::TensorDialect, lattigo::LattigoDialect>();
      });
}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
