#include "include/Target/OpenFhePke/OpenFhePkeEmitter.h"

#include <functional>
#include <string_view>

#include "include/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "include/Dialect/LWE/IR/LWEDialect.h"
#include "include/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "include/Dialect/Openfhe/IR/OpenfheOps.h"
#include "include/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "include/Target/OpenFhePke/OpenFheUtils.h"
#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "lib/Target/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
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

void registerToOpenFhePkeTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-openfhe-pke",
      "translate the openfhe dialect to C++ code against the OpenFHE pke API",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToOpenFhePke(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, openfhe::OpenfheDialect,
                        lwe::LWEDialect, polynomial::PolynomialDialect,
                        arith::ArithDialect, tensor::TensorDialect>();
      });
}

LogicalResult translateToOpenFhePke(Operation *op, llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  OpenFhePkeEmitter emitter(os, &variableNames);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult OpenFhePkeEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp>([&](auto op) { return printOperation(op); })
          // OpenFHE ops
          .Case<AddOp, SubOp, MulOp, MulPlainOp, SquareOp, NegateOp, MulConstOp,
                RelinOp, ModReduceOp, LevelReduceOp, RotOp, AutomorphOp,
                KeySwitchOp>([&](auto op) { return printOperation(op); })
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(func::FuncOp funcOp) {
  if (funcOp.getNumResults() != 1) {
    return funcOp.emitOpError() << "Only functions with a single return type "
                                   "are supported, but this function has "
                                << funcOp.getNumResults();
    return failure();
  }

  Type result = funcOp.getResultTypes()[0];
  if (failed(emitType(result))) {
    return funcOp.emitOpError() << "Failed to emit type " << result;
  }

  os << " " << funcOp.getName() << "(";
  os.indent();

  // Check the types without printing to enable failure outside of
  // commaSeparatedValues; maybe consider making commaSeparatedValues combine
  // the results into a FailureOr, like commaSeparatedTypes in tfhe_rust
  // emitter.
  for (Value arg : funcOp.getArguments()) {
    if (failed(convertType(arg.getType()))) {
      return funcOp.emitOpError() << "Failed to emit type " << arg.getType();
    }
  }

  os << commaSeparatedValues(funcOp.getArguments(), [&](Value value) {
    auto res = convertType(value.getType());
    return res.value() + " " + variableNames->getNameForValue(value);
  });
  os.unindent();
  os << ") {\n";
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

LogicalResult OpenFhePkeEmitter::printOperation(func::ReturnOp op) {
  if (op.getNumOperands() != 1) {
    op.emitError() << "Only one return value supported";
    return failure();
  }
  os << "return " << variableNames->getNameForValue(op.getOperands()[0])
     << ";\n";
  return success();
}

void OpenFhePkeEmitter::emitAssignPrefix(Value result) {
  os << "auto " << variableNames->getNameForValue(result) << " = ";
}

LogicalResult OpenFhePkeEmitter::printEvalMethod(
    ::mlir::Value result, ::mlir::Value cryptoContext,
    ::mlir::ValueRange nonEvalOperands, std::string_view op) {
  emitAssignPrefix(result);

  os << variableNames->getNameForValue(cryptoContext) << "->" << op << "(";
  os << commaSeparatedValues(nonEvalOperands, [&](Value value) {
    return variableNames->getNameForValue(value);
  });
  os << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(AddOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getLhs(), op.getRhs()}, "EvalAdd");
}

LogicalResult OpenFhePkeEmitter::printOperation(SubOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getLhs(), op.getRhs()}, "EvalSub");
}

LogicalResult OpenFhePkeEmitter::printOperation(MulOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getLhs(), op.getRhs()}, "EvalMult");
}

LogicalResult OpenFhePkeEmitter::printOperation(MulPlainOp op) {
  // OpenFHE defines an overload for EvalMult to work on both plaintext and
  // ciphertext inputs.
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext(), op.getPlaintext()}, "EvalMult");
}

LogicalResult OpenFhePkeEmitter::printOperation(MulConstOp op) {
  // OpenFHE defines an overload for EvalMult to work on constant inputs,
  // but only for some schemes.
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext(), op.getConstant()}, "EvalMult");
}

LogicalResult OpenFhePkeEmitter::printOperation(NegateOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext()}, "EvalNegate");
}

LogicalResult OpenFhePkeEmitter::printOperation(SquareOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext()}, "EvalSquare");
}

LogicalResult OpenFhePkeEmitter::printOperation(RelinOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext()}, "Relinearize");
}

LogicalResult OpenFhePkeEmitter::printOperation(ModReduceOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext()}, "ModReduce");
}

LogicalResult OpenFhePkeEmitter::printOperation(LevelReduceOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext()}, "LevelReduce");
}

LogicalResult OpenFhePkeEmitter::printOperation(RotOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext(), op.getIndex()}, "EvalRotate");
}

LogicalResult OpenFhePkeEmitter::printOperation(AutomorphOp op) {
  // EvalAutomorphism has a bit of a strange function signature in OpenFHE:
  //
  //     EvalAutomorphism(
  //       ConstCiphertext<DCRTPoly> ciphertext,
  //       int32_t i,
  //       const std::map<int32_t, EvalKey<DCRTPoly>>& evalKeyMap
  //     )
  //
  // Here i is an index to evalKeyMap, but no other data from evalKeyMap is
  // used. To match the API, we emit code that just creates a single-entry map
  // locally before calling EvalAutomorphism.
  //
  // This would probably be an easy upstream fix to add a specialized function
  // call if it becomes necessary.
  std::string mapName =
      variableNames->getNameForValue(op.getResult()) + "evalkeymap";
  auto result = convertType(op.getEvalKey().getType());
  os << "std::map<uint32_t, " << result << "> " << mapName << " = {{0, "
     << variableNames->getNameForValue(op.getEvalKey()) << "}};\n";

  emitAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getCryptoContext())
     << "->EvalAutomorphism(";
  os << variableNames->getNameForValue(op.getCiphertext()) << ", 0, " << mapName
     << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(KeySwitchOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext(), op.getEvalKey()}, "KeySwitch");
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  emitAssignPrefix(op.getResult());
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    os << intAttr.getValue() << ";\n";
  } else {
    return op.emitError() << "Unsupported constant type "
                          << valueAttr.getType();
  }
  return success();
}

LogicalResult OpenFhePkeEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

OpenFhePkeEmitter::OpenFhePkeEmitter(raw_ostream &os,
                                     SelectVariableNames *variableNames)
    : os(os), variableNames(variableNames) {}
}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
