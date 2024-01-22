#include "include/Target/TfheRustBool/TfheRustBoolEmitter.h"

#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <string_view>

#include "include/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolOps.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolTypes.h"
#include "lib/Target/TfheRustBool/TfheRustBoolTemplates.h"
#include "lib/Target/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
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
namespace tfhe_rust_bool {

void registerToTfheRustBoolTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-tfhe-rust-bool",
      "translate the bool tfhe_rs dialect to Rust code for boolean tfhe-rs",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToTfheRustBool(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, tfhe_rust_bool::TfheRustBoolDialect,
                        arith::ArithDialect, tensor::TensorDialect>();
      });
}

LogicalResult translateToTfheRustBool(Operation *op, llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  TfheRustBoolEmitter emitter(os, &variableNames);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult TfheRustBoolEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp>([&](auto op) { return printOperation(op); })
          // TfheRustBool ops
          .Case<AndOp, NandOp, OrOp, NorOp, XorOp, XnorOp>(
              [&](auto op) { return printOperation(op); })
          // Tensor ops
          .Case<tensor::ExtractOp, tensor::FromElementsOp>(
              [&](auto op) { return printOperation(op); })

          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult TfheRustBoolEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult TfheRustBoolEmitter::printOperation(func::FuncOp funcOp) {
  os << "pub fn " << funcOp.getName() << "(\n";
  os.indent();
  for (Value arg : funcOp.getArguments()) {
    auto argName = variableNames->getNameForValue(arg);
    os << argName << ": &";
    if (failed(emitType(arg.getType()))) {
      return funcOp.emitOpError()
             << "Failed to emit tfhe-rs bool type " << arg.getType();
    }
    os << ",\n";
  }
  os.unindent();
  os << ") -> ";

  if (funcOp.getNumResults() == 1) {
    Type result = funcOp.getResultTypes()[0];
    if (failed(emitType(result))) {
      return funcOp.emitOpError()
             << "Failed to emit tfhe-rs bool type " << result;
    }
  } else {
    auto result = commaSeparatedTypes(
        funcOp.getResultTypes(), [&](Type type) -> FailureOr<std::string> {
          auto result = convertType(type);
          if (failed(result)) {
            return funcOp.emitOpError()
                   << "Failed to emit tfhe-rs bool type " << type;
          }
          return result;
        });
    os << "(" << result.value() << ")";
  }

  os << " {\n";
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

LogicalResult TfheRustBoolEmitter::printOperation(func::ReturnOp op) {
  std::function<std::string(Value)> valueOrClonedValue = [&](Value value) {
    auto cloneStr = "";
    if (isa<BlockArgument>(value)) {
      cloneStr = ".clone()";
    }
    return variableNames->getNameForValue(value) + cloneStr;
  };

  if (op.getNumOperands() == 1) {
    os << valueOrClonedValue(op.getOperands()[0]) << "\n";
    return success();
  }

  os << "(" << commaSeparatedValues(op.getOperands(), valueOrClonedValue)
     << ")\n";
  return success();
}

void TfheRustBoolEmitter::emitAssignPrefix(Value result) {
  os << "let " << variableNames->getNameForValue(result) << " = ";
}

LogicalResult TfheRustBoolEmitter::printSksMethod(
    ::mlir::Value result, ::mlir::Value sks, ::mlir::ValueRange nonSksOperands,
    std::string_view op, SmallVector<std::string> operandTypes) {
  emitAssignPrefix(result);

  auto operandTypesIt = operandTypes.begin();
  os << variableNames->getNameForValue(sks) << "." << op << "(";
  os << commaSeparatedValues(nonSksOperands, [&](Value value) {
    return variableNames->getNameForValue(value) +
           (!operandTypes.empty() ? " as " + *operandTypesIt++ : "");
  });
  os << ");\n";
  return success();
}

LogicalResult TfheRustBoolEmitter::printOperation(CreateTrivialOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(), {op.getValue()},
                        "create_trivial", {"i1"});
}

LogicalResult TfheRustBoolEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  if (isa<IntegerType>(op.getType()) &&
      op.getType().getIntOrFloatBitWidth() == 1) {
    os << "let " << variableNames->getNameForValue(op.getResult())
       << " : bool = ";
    os << (cast<IntegerAttr>(valueAttr).getValue().isZero() ? "false" : "true")
       << ";\n";
    return success();
  }

  emitAssignPrefix(op.getResult());
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    os << intAttr.getValue() << ";\n";
  } else {
    return op.emitError() << "Unknown constant type " << valueAttr.getType();
  }
  return success();
}

LogicalResult TfheRustBoolEmitter::printOperation(tensor::ExtractOp op) {
  // We assume here that the indices are SSA values (not integer attributes).
  emitAssignPrefix(op.getResult());
  os << "&" << variableNames->getNameForValue(op.getTensor()) << "["
     << commaSeparatedValues(
            op.getIndices(),
            [&](Value value) { return variableNames->getNameForValue(value); })
     << "];\n";
  return success();
}

LogicalResult TfheRustBoolEmitter::printOperation(tensor::FromElementsOp op) {
  emitAssignPrefix(op.getResult());
  os << "vec![" << commaSeparatedValues(op.getOperands(), [&](Value value) {
    // Check if block argument, if so, clone.
    auto cloneStr = "";
    if (isa<BlockArgument>(value)) {
      cloneStr = ".clone()";
    }
    return variableNames->getNameForValue(value) + cloneStr;
  }) << "];\n";
  return success();
}

LogicalResult TfheRustBoolEmitter::printOperation(AndOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getLhs(), op.getRhs()}, "and");
}

LogicalResult TfheRustBoolEmitter::printOperation(NandOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getLhs(), op.getRhs()}, "nand");
}

LogicalResult TfheRustBoolEmitter::printOperation(OrOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getLhs(), op.getRhs()}, "or");
}

LogicalResult TfheRustBoolEmitter::printOperation(NorOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getLhs(), op.getRhs()}, "nor");
}

LogicalResult TfheRustBoolEmitter::printOperation(XorOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getLhs(), op.getRhs()}, "xor");
}

LogicalResult TfheRustBoolEmitter::printOperation(XnorOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getLhs(), op.getRhs()}, "xnor");
}

FailureOr<std::string> TfheRustBoolEmitter::convertType(Type type) {
  // Note: these are probably not the right type names to use exactly, and they
  // will need to chance to the right values once we try to compile it against
  // a specific API version.
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    // A lambda in a type switch statement can't return multiple types.
    // FIXME: why can't both types be FailureOr<std::string>?
    auto elementTy = convertType(shapedType.getElementType());
    if (failed(elementTy)) return failure();
    return std::string("Vec<" + elementTy.value() + ">");
  }
  return llvm::TypeSwitch<Type &, FailureOr<std::string>>(type)
      .Case<EncryptedBoolType>(
          [&](auto type) { return std::string("Ciphertext"); })
      .Case<ServerKeyType>([&](auto type) { return std::string("ServerKey"); })
      .Default([&](Type &) { return failure(); });
}

LogicalResult TfheRustBoolEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

TfheRustBoolEmitter::TfheRustBoolEmitter(raw_ostream &os,
                                         SelectVariableNames *variableNames)
    : os(os), variableNames(variableNames) {}
}  // namespace tfhe_rust_bool
}  // namespace heir
}  // namespace mlir
