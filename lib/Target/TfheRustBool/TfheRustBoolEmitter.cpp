#include "lib/Target/TfheRustBool/TfheRustBoolEmitter.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOps.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolTypes.h"
#include "lib/Target/TfheRust/Utils.h"
#include "lib/Target/TfheRustBool/TfheRustBoolTemplates.h"
#include "lib/Target/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
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

namespace {

// getRustIntegerType returns the width of the closest builtin integer type.
FailureOr<int> getRustIntegerType(int width) {
  for (int candidate : {8, 16, 32, 64, 128}) {
    if (width <= candidate) {
      return candidate;
    }
  }
  return failure();
}

}  // namespace

void registerToTfheRustBoolTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-tfhe-rust-bool",
      "translate the bool tfhe_rs dialect to Rust code for boolean tfhe-rs",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToTfheRustBool(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, tfhe_rust_bool::TfheRustBoolDialect,
                        affine::AffineDialect, arith::ArithDialect,
                        tensor::TensorDialect, memref::MemRefDialect>();
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
          // Affine ops
          .Case<affine::AffineForOp, affine::AffineYieldOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp, arith::IndexCastOp, arith::ShRSIOp,
                arith::ShLIOp, arith::TruncIOp, arith::AndIOp>(
              [&](auto op) { return printOperation(op); })
          // MemRef ops
          .Case<memref::AllocOp, memref::LoadOp, memref::StoreOp>(
              [&](auto op) { return printOperation(op); })
          // TfheRustBool ops
          .Case<AndOp, NandOp, OrOp, NorOp, NotOp, XorOp, XnorOp,
                CreateTrivialOp>([&](auto op) { return printOperation(op); })
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
  if (failed(tfhe_rust::canEmitFuncForTfheRust(funcOp))) {
    // Return success implies print nothing, and note the called function
    // emits a warning.
    return success();
  }

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
    auto suffix = "";
    if (isa<BlockArgument>(value)) {
      suffix = ".clone()";
    }
    if (isa<tensor::FromElementsOp>(value.getDefiningOp())) {
      suffix = ".into_iter().cloned().collect()";
    }
    if (isa<memref::AllocOp>(value.getDefiningOp())) {
      // MemRefs (BTreeMap<(usize, ...), Ciphertext>) must be converted to
      // Vec<Ciphertext>
      suffix = ".into_values().collect()";
    };
    return variableNames->getNameForValue(value) + suffix;
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

void TfheRustBoolEmitter::emitReferenceConversion(Value value) {
  auto tensorType = dyn_cast<TensorType>(value.getType());

  if (isa<EncryptedBoolType>(tensorType.getElementType())) {
    auto varName = variableNames->getNameForValue(value);

    os << "let " << varName << "_ref = " << varName << ".clone();\n";
    os << "let " << varName << "_ref: Vec<&Ciphertext> = " << varName
       << ".iter().collect();\n";
  }
}

LogicalResult TfheRustBoolEmitter::printSksMethod(
    ::mlir::Value result, ::mlir::Value sks, ::mlir::ValueRange nonSksOperands,
    std::string_view op, SmallVector<std::string> operandTypes) {
  if (isa<TensorType>(nonSksOperands[0].getType())) {
    auto *opParent = nonSksOperands[0].getDefiningOp();

    if (!opParent) {
      for (auto nonSksOperand : nonSksOperands) {
        emitReferenceConversion(nonSksOperand);
      }
    }

    emitAssignPrefix(result);

    os << variableNames->getNameForValue(sks) << "." << op << "_packed(";
    os << commaSeparatedValues(nonSksOperands, [&](Value value) {
      auto *prefix = "&";
      auto suffix = "";
      // First check if a DefiningOp exists
      // if not: comes from function definition
      mlir::Operation *opParent = value.getDefiningOp();
      if (opParent) {
        if (!isa<tensor::FromElementsOp>(value.getDefiningOp()) &&
            !isa<tensor::ExtractOp>(opParent))
          prefix = "";

      } else {
        prefix = "&";
        suffix = "_ref";
      }

      return prefix + variableNames->getNameForValue(value) + suffix;
    });
    os << ");\n";
    return success();

  } else {
    emitAssignPrefix(result);

    auto operandTypesIt = operandTypes.begin();
    os << variableNames->getNameForValue(sks) << "." << op << "(";
    os << commaSeparatedValues(nonSksOperands, [&](Value value) {
      auto *prefix = value.getType().hasTrait<PassByReference>() ? "&" : "";
      // First check if a DefiningOp exists
      // if not: comes from function definition
      mlir::Operation *op = value.getDefiningOp();
      if (op) {
        auto reference_predicate =
            isa<tensor::ExtractOp>(op) || isa<memref::LoadOp>(op);
        prefix = reference_predicate ? "" : prefix;
      } else {
        prefix = "";
      }

      return prefix + variableNames->getNameForValue(value) +
             (!operandTypes.empty() ? " as " + *operandTypesIt++ : "");
    });
    os << ");\n";
    return success();
  }
}

LogicalResult TfheRustBoolEmitter::printOperation(CreateTrivialOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(), {op.getValue()},
                        "trivial_encrypt", {"bool"});
}

LogicalResult TfheRustBoolEmitter::printOperation(affine::AffineForOp op) {
  if (op.getStepAsInt() > 1) {
    return op.emitOpError() << "AffineForOp has step > 1";
  }
  os << "for " << variableNames->getNameForValue(op.getInductionVar()) << " in "
     << op.getConstantLowerBound() << ".." << op.getConstantUpperBound()
     << " {\n";
  os.indent();

  auto res = op.getBody()->walk([&](Operation *op) {
    if (failed(translate(*op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) {
    return failure();
  }

  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult TfheRustBoolEmitter::printOperation(affine::AffineYieldOp op) {
  if (op->getNumResults() != 0) {
    return op.emitOpError() << "AffineYieldOp has non-zero number of results";
  }
  return success();
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

LogicalResult TfheRustBoolEmitter::printOperation(arith::IndexCastOp op) {
  emitAssignPrefix(op.getOut());
  os << variableNames->getNameForValue(op.getIn()) << " as ";
  if (failed(emitType(op.getOut().getType()))) {
    return op.emitOpError()
           << "Failed to emit index cast type " << op.getOut().getType();
  }
  os << ";\n";
  return success();
}

LogicalResult TfheRustBoolEmitter::printBinaryOp(::mlir::Value result,
                                                 ::mlir::Value lhs,
                                                 ::mlir::Value rhs,
                                                 std::string_view op) {
  emitAssignPrefix(result);
  os << variableNames->getNameForValue(lhs) << " " << op << " "
     << variableNames->getNameForValue(rhs) << ";\n";
  return success();
}

LogicalResult TfheRustBoolEmitter::printOperation(::mlir::arith::ShLIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "<<");
}

LogicalResult TfheRustBoolEmitter::printOperation(::mlir::arith::AndIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "&");
}

LogicalResult TfheRustBoolEmitter::printOperation(::mlir::arith::ShRSIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), ">>");
}

LogicalResult TfheRustBoolEmitter::printOperation(::mlir::arith::TruncIOp op) {
  emitAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getIn());
  if (isa<IntegerType>(op.getType()) &&
      op.getType().getIntOrFloatBitWidth() == 1) {
    // Compare with zero to truncate to a boolean.
    os << " != 0";
  } else {
    os << " as ";
    if (failed(emitType(op.getType()))) {
      return op.emitOpError()
             << "Failed to emit truncated type " << op.getType();
    }
  }
  os << ";\n";
  return success();
}

// Use a BTreeMap<(usize, ...), Ciphertext>.
LogicalResult TfheRustBoolEmitter::printOperation(memref::AllocOp op) {
  os << "let mut " << variableNames->getNameForValue(op.getMemref())
     << " : BTreeMap<("
     << std::accumulate(
            std::next(op.getMemref().getType().getShape().begin()),
            op.getMemref().getType().getShape().end(), std::string("usize"),
            [&](std::string a, int64_t value) { return a + ", usize"; })
     << "), ";
  if (failed(emitType(op.getMemref().getType().getElementType()))) {
    return op.emitOpError() << "Failed to get memref element type";
  }

  os << "> = BTreeMap::new();\n";
  return success();
}

// Store into a BTreeMap<(usize, ...), Ciphertext>
LogicalResult TfheRustBoolEmitter::printOperation(memref::StoreOp op) {
  // We assume here that the indices are SSA values (not integer attributes).
  os << variableNames->getNameForValue(op.getMemref());
  os << ".insert((" << commaSeparatedValues(op.getIndices(), [&](Value value) {
    return variableNames->getNameForValue(value) + std::string(" as usize");
  }) << "), ";

  // Note: we may not need to clone all the time, but the BTreeMap stores
  // Ciphertexts, not &Ciphertexts. This is because results computed inside for
  // loops will not live long enough.
  auto suffix = ".clone()";
  os << variableNames->getNameForValue(op.getValueToStore()) << suffix
     << ");\n";
  return success();
}

// Produces a &Ciphertext
LogicalResult TfheRustBoolEmitter::printOperation(memref::LoadOp op) {
  // We assume here that the indices are SSA values (not integer attributes).
  if (isa<BlockArgument>(op.getMemref())) {
    emitAssignPrefix(op.getResult());
    os << "&" << variableNames->getNameForValue(op.getMemRef()) << "["
       << flattenIndexExpression(op.getMemRefType(), op.getIndices(),
                                 [&](Value value) {
                                   return variableNames->getNameForValue(value);
                                 })
       << "];\n";
    return success();
  }

  // Treat this as a BTreeMap
  emitAssignPrefix(op.getResult());
  os << "&" << variableNames->getNameForValue(op.getMemref()) << ".get(&("
     << commaSeparatedValues(op.getIndices(),
                             [&](Value value) {
                               return variableNames->getNameForValue(value) +
                                      " as usize";
                             })
     << ")).unwrap();\n";
  return success();
}

// Produces a &Ciphertext
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

// Need to produce a Vec<&Ciphertext>
LogicalResult TfheRustBoolEmitter::printOperation(tensor::FromElementsOp op) {
  emitAssignPrefix(op.getResult());
  os << "vec![" << commaSeparatedValues(op.getOperands(), [&](Value value) {
    // Check if block argument, if so, clone.
    auto cloneStr = isa<BlockArgument>(value) ? ".clone()" : "";
    // Get the name of defining operation its dialect
    auto tfhe_op =
        value.getDefiningOp()->getDialect()->getNamespace() == "tfhe_rust_bool";
    auto prefix = tfhe_op ? "&" : "";
    return std::string(prefix) + variableNames->getNameForValue(value) +
           cloneStr;
  }) << "];\n";
  return success();
}

LogicalResult TfheRustBoolEmitter::printOperation(NotOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(), {op.getInput()},
                        "not");
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
  // Note: these are probably not the right type names to use exactly, and
  // they will need to chance to the right values once we try to compile it
  // against a specific API version.
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    // A lambda in a type switch statement can't return multiple types.
    // FIXME: why can't both types be FailureOr<std::string>?
    auto elementTy = convertType(shapedType.getElementType());
    if (failed(elementTy)) return failure();

    return std::string(std::string("Vec<") + elementTy.value() + ">");
  }
  return llvm::TypeSwitch<Type &, FailureOr<std::string>>(type)
      .Case<EncryptedBoolType>(
          [&](auto type) { return std::string("Ciphertext"); })
      .Case<ServerKeyType>([&](auto type) { return std::string("ServerKey"); })
      .Case<IntegerType>([&](IntegerType type) -> FailureOr<std::string> {
        if (type.getWidth() == 1) {
          return std::string("bool");
        }
        auto width = getRustIntegerType(type.getWidth());
        if (failed(width)) return failure();
        return (type.isUnsigned() ? std::string("u") : "") + "i" +
               std::to_string(width.value());
      })
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
