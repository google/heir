#include "include/Target/TfheRust/TfheRustEmitter.h"

#include <numeric>
#include <string_view>

#include "include/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "include/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "include/Dialect/TfheRust/IR/TfheRustOps.h"
#include "include/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "lib/Target/TfheRust/TfheRustTemplates.h"
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tfhe_rust {

// TODO(https://github.com/google/heir/issues/230): Have a separate pass that
// topo-sorts the gate ops into levels, and use scf.parallel_for to represent
// them.

void registerToTfheRustTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-tfhe-rust",
      "translate the tfhe_rs dialect to Rust code for tfhe-rs",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToTfheRust(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, tfhe_rust::TfheRustDialect,
                        arith::ArithDialect>();
      });
}

LogicalResult translateToTfheRust(Operation *op, llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  TfheRustEmitter emitter(os, &variableNames);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult TfheRustEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp>([&](auto op) { return printOperation(op); })
          // TfheRust ops
          .Case<AddOp, ApplyLookupTableOp, ScalarLeftShiftOp>(
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

std::string commaSeparatedValues(
    ValueRange values, std::function<std::string(Value)> valueToString) {
  return std::accumulate(
      std::next(values.begin()), values.end(), valueToString(values[0]),
      [&](std::string &a, Value b) { return a + ", " + valueToString(b); });
}

FailureOr<std::string> commaSeparatedTypes(
    TypeRange types, std::function<FailureOr<std::string>(Type)> typeToString) {
  return std::accumulate(
      std::next(types.begin()), types.end(), typeToString(types[0]),
      [&](FailureOr<std::string> &a, Type b) -> FailureOr<std::string> {
        auto result = typeToString(b);
        if (failed(result)) {
          return failure();
        }
        return a.value() + ", " + result.value();
      });
}

LogicalResult TfheRustEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult TfheRustEmitter::printOperation(func::FuncOp funcOp) {
  os << "pub fn " << funcOp.getName() << "(\n";
  os.indent();
  for (Value arg : funcOp.getArguments()) {
    auto argName = variableNames->getNameForValue(arg);
    os << argName << ": &";
    if (failed(emitType(arg.getType()))) {
      return funcOp.emitOpError()
             << "Failed to emit thfe-rs type " << arg.getType();
    }
    os << ",\n";
  }
  os.unindent();
  os << ") -> ";

  if (funcOp.getNumResults() == 1) {
    Type result = funcOp.getResultTypes()[0];
    if (failed(emitType(result))) {
      return funcOp.emitOpError() << "Failed to emit thfe-rs type " << result;
    }
  } else {
    auto result = commaSeparatedTypes(
        funcOp.getResultTypes(), [&](Type type) -> FailureOr<std::string> {
          auto result = convertType(type);
          if (failed(result)) {
            return funcOp.emitOpError()
                   << "Failed to emit thfe-rs type " << type;
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

LogicalResult TfheRustEmitter::printOperation(func::ReturnOp op) {
  if (op.getNumOperands() == 1) {
    os << variableNames->getNameForValue(op.getOperands()[0]) << "\n";
    return success();
  }

  os << "(" << commaSeparatedValues(op.getOperands(), [&](Value value) {
    return variableNames->getNameForValue(value);
  }) << ")\n";
  return success();
}

void TfheRustEmitter::emitAssignPrefix(Value result) {
  os << "let " << variableNames->getNameForValue(result) << " = ";
}

LogicalResult TfheRustEmitter::printSksMethod(::mlir::Value result,
                                              ::mlir::Value sks,
                                              ::mlir::ValueRange nonSksOperands,
                                              std::string_view op) {
  emitAssignPrefix(result);
  os << variableNames->getNameForValue(sks) << "." << op << "(";
  os << commaSeparatedValues(nonSksOperands, [&](Value value) {
    return "&" + variableNames->getNameForValue(value);
  });
  os << ");\n";
  return success();
}

LogicalResult TfheRustEmitter::printOperation(AddOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getLhs(), op.getRhs()}, "unchecked_add");
}

LogicalResult TfheRustEmitter::printOperation(ApplyLookupTableOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getInput(), op.getLookupTable()},
                        "apply_lookup_table");
}

LogicalResult TfheRustEmitter::printOperation(ScalarLeftShiftOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getCiphertext(), op.getShiftAmount()},
                        "scalar_left_shift");
}

LogicalResult TfheRustEmitter::printOperation(arith::ConstantOp op) {
  emitAssignPrefix(op.getResult());
  auto valueAttr = op.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    os << intAttr.getValue() << ";\n";
  } else {
    return op.emitError() << "Unknown constant type " << valueAttr.getType();
  }
  return success();
}

FailureOr<std::string> TfheRustEmitter::convertType(Type type) {
  // Note: these are probably not the right type names to use exactly, and they
  // will need to chance to the right values once we try to compile it against
  // a specific API version.
  return llvm::TypeSwitch<Type &, FailureOr<std::string>>(type)
      .Case<EncryptedUInt3Type>(
          [&](auto type) { return std::string("Ciphertext"); })
      .Case<ServerKeyType>([&](auto type) { return std::string("ServerKey"); })
      .Case<LookupTableType>(
          [&](auto type) { return std::string("LookupTableOwned"); })
      .Default([&](Type &) { return failure(); });
}

LogicalResult TfheRustEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir
