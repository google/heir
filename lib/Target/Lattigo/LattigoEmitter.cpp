#include "lib/Target/Lattigo/LattigoEmitter.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Target/Lattigo/LattigoTemplates.h"
#include "lib/Utils/TargetUtils/TargetUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

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
          .Case<func::FuncOp, func::ReturnOp>(
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
  // if (funcOp.getNumResults() != 1) {
  //   return emitError(funcOp.getLoc(),
  //                    llvm::formatv("Only functions with a single return type
  //                    "
  //                                  "are supported, but this function has ",
  //                                  funcOp.getNumResults()));
  //   return failure();
  // }

  os << "func " << funcOp.getName() << "() {\n";
  os.indent();

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
  // if (op.getNumOperands() != 1) {
  //   return emitError(op.getLoc(), "Only one return value supported");
  // }
  // os << "return " << variableNames->getNameForValue(op.getOperands()[0])
  //    << ";\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  std::string valueString;
  auto res = llvm::TypeSwitch<Attribute, LogicalResult>(valueAttr)
                 .Case<IntegerAttr>([&](IntegerAttr intAttr) {
                   valueString =
                       "[]int64{" + std::to_string(intAttr.getInt()) + "}";
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
  // return printNewMethod(op.getResult(), {op.getParams()}, "bgv.NewEvaluator",
  //                       false);
  os << getName(op.getResult()) << " := " << "bgv.NewEvaluator(";
  os << getName(op.getParams()) << ", nil)\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVNewPlaintextOp op) {
  // return printNewMethod(op.getResult(), {op.getParams()}, "bgv.NewPlaintext",
  //                       false);
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

LogicalResult LattigoEmitter::printOperation(BGVNewParametersFromLiteralOp op) {
  std::string errResult = ", err";
  os << getName(op.getResult()) << errResult
     << " := bgv.NewParametersFromLiteral(";
  os << "bgv.ParametersLiteral{";
  os.indent();
  os << "LogN: " << op.getParamsLiteral().getLogN() << ",\n";
  os << "LogQ: " << printDenseI32ArrayAttr(op.getParamsLiteral().getLogQ())
     << ",\n";
  os << "LogP: " << printDenseI32ArrayAttr(op.getParamsLiteral().getLogP())
     << ",\n";
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
  // workaround
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
                        lattigo::LattigoDialect>();
      });
}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
