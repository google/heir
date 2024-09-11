#include "lib/Target/Jaxite/JaxiteEmitter.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Jaxite/IR/JaxiteDialect.h"
#include "lib/Dialect/Jaxite/IR/JaxiteOps.h"
#include "lib/Dialect/Jaxite/IR/JaxiteTypes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Target/Jaxite/JaxiteTemplates.h"
#include "lib/Target/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
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
namespace jaxite {

void registerToJaxiteTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-jaxite", "translate the jaxite dialect to python code for jaxite",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToJaxite(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, jaxite::JaxiteDialect,
                        arith::ArithDialect, tensor::TensorDialect,
                        lwe::LWEDialect, memref::MemRefDialect>();
      });
}

LogicalResult translateToJaxite(Operation *op, llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  JaxiteEmitter emitter(os, &variableNames);
  return emitter.translate(*op);
}

LogicalResult JaxiteEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Jaxite ops
          .Case<Lut3Op, ConstantOp>([&](auto op) { return printOperation(op); })
          // Tensor ops
          .Case<tensor::ExtractOp, tensor::FromElementsOp>(
              [&](auto op) { return printOperation(op); })
          // Memref ops
          .Case<memref::LoadOp, memref::StoreOp, memref::AllocOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp>([&](auto op) { return success(); })
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult JaxiteEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult JaxiteEmitter::printOperation(func::FuncOp funcOp) {
  os << "def " << funcOp.getName() << "(\n";
  os.indent();
  for (Value arg : funcOp.getArguments()) {
    auto argName = variableNames->getNameForValue(arg);
    os << argName << ": ";
    if (failed(emitType(arg.getType()))) {
      return funcOp.emitOpError()
             << "Failed to emit jaxite type " << arg.getType();
    }
    os << ",\n";
    if (isa<jaxite::ServerKeySetType>(arg.getType())) {
      serverKeySetArg_ = argName;
    }
    if (isa<jaxite::ParamsType>(arg.getType())) {
      paramsArg_ = argName;
    }
  }
  os.unindent();
  os << ")";

  if (serverKeySetArg_.empty() || paramsArg_.empty()) {
    return funcOp.emitWarning() << "Missing server keyset or params";
  }

  if (funcOp.getNumResults() > 0) {
    os << " -> ";
    if (funcOp.getNumResults() == 1) {
      Type result = funcOp.getResultTypes()[0];
      if (failed(emitType(result))) {
        return funcOp.emitOpError() << "Failed to emit jaxite type " << result;
      }
    } else {
      auto result = commaSeparatedTypes(
          funcOp.getResultTypes(), [&](Type type) -> FailureOr<std::string> {
            auto result = convertType(type);
            if (failed(result)) {
              return funcOp.emitOpError()
                     << "Failed to emit jaxite type " << type;
            }
            return result;
          });
      os << "(" << result.value() << ")";
    }
  }

  os << ":\n";
  os.indent();

  os << "temp_nodes: Dict[int, Any] = {}" << "\n";

  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }

  os.unindent();
  os << "\n";
  return success();
}

LogicalResult JaxiteEmitter::printOperation(func::ReturnOp op) {
  std::function<std::string(Value)> resultValue = [&](Value value) {
    if (isa<BlockArgument>(value)) {
      // Function arguments used as outputs.
      return variableNames->getNameForValue(value);
    } else {
      return "temp_nodes[" +
             std::to_string(variableNames->getIntForValue(value)) + "]";
    }
  };
  if (op.getNumOperands() == 0) {
    return success();
  }
  if (op.getNumOperands() == 1) {
    os << "return " << resultValue(op.getOperands()[0]) << "\n";
    return success();
  } else {
    os << "return (" << commaSeparatedValues(op.getOperands(), resultValue)
       << ")\n";
    return success();
  }
  return failure();
}

LogicalResult JaxiteEmitter::printOperation(Lut3Op op) {
  emitAssignPrefix(op.getResult());
  SmallString<16> int_str;
  cast<IntegerAttr>(
      dyn_cast<arith::ConstantOp>(op.getTruthTable().getDefiningOp())
          .getValue())
      .getValue()
      .toStringUnsigned(int_str);
  os << "jaxite_bool.lut3(" << "temp_nodes["
     << variableNames->getIntForValue(op.getA()) << "], " << "temp_nodes["
     << variableNames->getIntForValue(op.getB()) << "], " << "temp_nodes["
     << variableNames->getIntForValue(op.getC()) << "], " << int_str << ", "
     << serverKeySetArg_ << ", " << paramsArg_ << ")\n";
  return success();
}

LogicalResult JaxiteEmitter::printOperation(ConstantOp op) {
  emitAssignPrefix(op.getResult());
  os << "jaxite_bool.constant(";
  auto bool_constant_op =
      dyn_cast<arith::ConstantOp>(op.getValue().getDefiningOp());
  os << (dyn_cast<IntegerAttr>(bool_constant_op.getValue()).getValue().isZero()
             ? "False"
             : "True");
  os << ", " << paramsArg_ << ")\n";
  return success();
}

void JaxiteEmitter::emitAssignPrefix(Value result) {
  os << "temp_nodes[" << variableNames->getIntForValue(result) << "] = ";
}

LogicalResult JaxiteEmitter::printOperation(tensor::ExtractOp op) {
  emitAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getTensor()) << "["
     << dyn_cast<IntegerAttr>(
            dyn_cast<arith::ConstantOp>(op.getIndices()[0].getDefiningOp())
                .getValue())
            .getValue()
     << "]\n";
  return success();
}

LogicalResult JaxiteEmitter::printOperation(tensor::FromElementsOp op) {
  emitAssignPrefix(op.getResult());
  os << "[" << commaSeparatedValues(op.getOperands(), [&](Value value) {
    return "temp_nodes[" +
           std::to_string(variableNames->getIntForValue(value)) + "]";
  }) << "]\n";
  return success();
}

// Loading variables.
// Example: temp_nodes[idx] = input[i]
LogicalResult JaxiteEmitter::printOperation(memref::LoadOp op) {
  emitAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getMemref());
  os << bracketEnclosedValues(op.getIndices(), [&](Value value) {
    SmallString<16> idx_str;
    dyn_cast<IntegerAttr>(
        dyn_cast<arith::ConstantOp>(value.getDefiningOp()).getValue())
        .getValue()
        .toStringUnsigned(idx_str);
    return std::string(idx_str);
  });
  os << "\n";
  return success();
}

// memref::AllocOp initializes a variable of a specific shape. Translation in
// Jaxite is to allocate multidimensional array of the same shape.
// Example: temp_nodes[idx] = np.full((ixj), None)
// Note: memref::AllocOp and memref::StoreOp need to be in sync on how the
// indices are processed
LogicalResult JaxiteEmitter::printOperation(memref::AllocOp op) {
  emitAssignPrefix(op.getResult());
  os << "np.full(("
     << std::accumulate(std::next(op.getMemref().getType().getShape().begin()),
                        op.getMemref().getType().getShape().end(),
                        std::to_string(op.getMemref().getType().getShape()[0]),
                        [&](const std::string &a, int64_t b) {
                          return a + ", " + std::to_string(b);
                        })
     << "), None)";
  os << "\n";
  return success();
}

// Assuming StoreOp is only used while storing results.
// Example: temp_nodes[result_idx][idx] = temp_nodes[i]
// Note: memref::AllocOp and memref::StoreOp need to be in sync on how the
// indices are processed.
LogicalResult JaxiteEmitter::printOperation(memref::StoreOp op) {
  os << "temp_nodes[" << variableNames->getIntForValue(op.getMemref()) << "]";
  os << bracketEnclosedValues(op.getIndices(), [&](Value value) {
    SmallString<16> idx_str;
    dyn_cast<IntegerAttr>(
        dyn_cast<arith::ConstantOp>(value.getDefiningOp()).getValue())
        .getValue()
        .toStringUnsigned(idx_str);
    return std::string(idx_str);
  });
  os << " = " << "temp_nodes["
     << variableNames->getIntForValue(op.getValueToStore()) << "]";
  os << "\n";
  return success();
}

FailureOr<std::string> JaxiteEmitter::convertType(Type type) {
  // Note: these are probably not the right type names to use exactly, and
  // they will need to change to the right values once we try to compile it
  // against a specific API version.
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    // A lambda in a type switch statement can't return multiple types.
    // FIXME: why can't both types be FailureOr<std::string>?
    auto elementTy = convertType(shapedType.getElementType());
    if (failed(elementTy)) return failure();

    return std::string(std::string("list[") + elementTy.value() + "]");
  }
  return llvm::TypeSwitch<Type &, FailureOr<std::string>>(type)
      .Case<lwe::LWECiphertextType>(
          [&](auto type) { return std::string("types.LweCiphertext"); })
      .Case<ServerKeySetType>(
          [&](auto type) { return std::string("jaxite_bool.ServerKeySet"); })
      .Case<ParamsType>(
          [&](auto type) { return std::string("jaxite_bool.Parameters"); })
      .Default([&](Type &) { return failure(); });
}

LogicalResult JaxiteEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

JaxiteEmitter::JaxiteEmitter(raw_ostream &os,
                             SelectVariableNames *variableNames)
    : os(os), variableNames(variableNames) {}

}  // namespace jaxite
}  // namespace heir
}  // namespace mlir
