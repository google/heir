#include "lib/Target/JaxiteWord/JaxiteWordEmitter.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordTypes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Target/JaxiteWord/JaxiteWordTemplates.h"
#include "lib/Utils/TargetUtils.h"
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
namespace jaxiteword {

void registerToJaxiteWordTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-jaxiteword",
      "translate the JaxiteWord dialect to python code for jaxiteword",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToJaxiteWord(op, output);
      },
      [](DialectRegistry& registry) {
        registry.insert<func::FuncDialect, jaxiteword::JaxiteWordDialect,
                        arith::ArithDialect, tensor::TensorDialect,
                        lwe::LWEDialect, memref::MemRefDialect>();
      });
}

LogicalResult translateToJaxiteWord(Operation* op, llvm::raw_ostream& os) {
  SelectVariableNames variableNames(op);
  JaxiteWordEmitter emitter(os, &variableNames);
  return emitter.translate(*op);
}

LogicalResult JaxiteWordEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // JaxiteWord ops
          .Case<AddOp>([&](auto op) { return printOperation(op); })
          // Tensor ops
          .Case<tensor::ExtractOp, tensor::FromElementsOp>(
              [&](auto op) { return printOperation(op); })
          // Memref ops
          .Case<memref::LoadOp, memref::StoreOp, memref::AllocOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp>([&](auto op) { return success(); })
          .Default([&](Operation&) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(func::FuncOp funcOp) {
  os << "def " << funcOp.getName() << "(\n";
  os.indent();
  for (Value arg : funcOp.getArguments()) {
    auto argName = variableNames->getNameForValue(arg);
    os << argName << ": ";
    if (failed(emitType(arg.getType()))) {
      return funcOp.emitOpError()
             << "Failed to emit JaxiteWord type " << arg.getType();
    }
    os << ",\n";
    if (isa<jaxiteword::CiphertextType>(arg.getType())) {
      CiphertextArg_ = argName;
    }
    if (isa<jaxiteword::ModulusListType>(arg.getType())) {
      ModulusListArg_ = argName;
    }
  }
  os.unindent();
  os << ")";

  if (CiphertextArg_.empty() || ModulusListArg_.empty()) {
    return funcOp.emitWarning() << "Missing server keyset or ModulusList";
  }

  if (funcOp.getNumResults() > 0) {
    os << " -> ";
    if (funcOp.getNumResults() == 1) {
      Type result = funcOp.getResultTypes()[0];
      if (failed(emitType(result))) {
        return funcOp.emitOpError()
               << "Failed to emit JaxiteWord type " << result;
      }
    } else {
      auto result = commaSeparatedTypes(
          funcOp.getResultTypes(), [&](Type type) -> FailureOr<std::string> {
            auto result = convertType(type);
            if (failed(result)) {
              return funcOp.emitOpError()
                     << "Failed to emit JaxiteWord type " << type;
            }
            return result;
          });
      os << "(" << result.value() << ")";
    }
  }

  os << ":\n";
  os.indent();

  for (Block& block : funcOp.getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }

  os.unindent();
  os << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(func::ReturnOp op) {
  std::function<std::string(Value)> resultValue = [&](Value value) {
    if (isa<BlockArgument>(value)) {
      // Function arguments used as outputs.
      return variableNames->getNameForValue(value);
    } else {
      return "v" + std::to_string(variableNames->getIntForValue(value));
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

LogicalResult JaxiteWordEmitter::printOperation(AddOp op) {
  emitAssignPrefix(op.getResult());
  os << op.getOperationName() << "(" << "v"
     << variableNames->getIntForValue(op.getValueA()) << ", " << "v"
     << variableNames->getIntForValue(op.getValueB()) << ", " << "v"
     << variableNames->getIntForValue(op.getModulusList()) << ")\n";
  return success();
}

void JaxiteWordEmitter::emitAssignPrefix(Value result) {
  os << "v" << variableNames->getIntForValue(result) << " = ";
}

LogicalResult JaxiteWordEmitter::printOperation(tensor::ExtractOp op) {
  emitAssignPrefix(op.getResult());
  if (isa<BlockArgument>(op.getTensor())) {
    os << variableNames->getNameForValue(op.getTensor());
  } else {
    os << "temp_nodes[" << variableNames->getIntForValue(op.getTensor()) << "]";
  }
  os << "["
     << dyn_cast<IntegerAttr>(
            dyn_cast<arith::ConstantOp>(op.getIndices()[0].getDefiningOp())
                .getValue())
            .getValue()
     << "]\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(tensor::FromElementsOp op) {
  if (op.getNumOperands() == 0) {
    return success();
  }
  if (isa<jaxiteword::AddOp>(op->getOperands()[0].getDefiningOp())) {
    return success();
  }
  emitAssignPrefix(op.getResult());
  os << "[" << commaSeparatedValues(op.getOperands(), [&](Value value) {
    return "temp_nodes[" +
           std::to_string(variableNames->getIntForValue(value)) + "]";
  }) << "]\n";
  return success();
}

// Loading variables.
// Example: temp_nodes[idx] = input[i]
LogicalResult JaxiteWordEmitter::printOperation(memref::LoadOp op) {
  emitAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getMemref());
  if (isa<BlockArgument>(op.getMemref())) {
    // We assume the arguments to the function are flattened.
    // We assume here that the indices are SSA values (not integer attributes).
    os << "["
       << flattenedIndex(
              op.getMemRefType(), op.getIndices(),
              [&](Value value) {
                return dyn_cast<IntegerAttr>(
                           dyn_cast<arith::ConstantOp>(value.getDefiningOp())
                               .getValue())
                    .getValue()
                    .getSExtValue();
              })
       << "]";
  } else {
    os << bracketEnclosedValues(op.getIndices(), [&](Value value) {
      SmallString<16> idx_str;
      dyn_cast<IntegerAttr>(
          dyn_cast<arith::ConstantOp>(value.getDefiningOp()).getValue())
          .getValue()
          .toStringUnsigned(idx_str);
      return std::string(idx_str);
    });
  }
  os << "\n";
  return success();
}

// memref::AllocOp initializes a variable of a specific shape. Translation in
// JaxiteWord is to allocate a flattened array.
// Example: temp_nodes[idx] = jnp.full((ixj), None)
// Note: memref::AllocOp and memref::StoreOp need to be in sync on how the
// indices are processed
LogicalResult JaxiteWordEmitter::printOperation(memref::AllocOp op) {
  emitAssignPrefix(op.getResult());
  os << "jnp.full(("
     << std::accumulate(std::next(op.getMemref().getType().getShape().begin()),
                        op.getMemref().getType().getShape().end(),
                        std::to_string(op.getMemref().getType().getShape()[0]),
                        [&](const std::string& a, int64_t b) {
                          return a + "*" + std::to_string(b);
                        })
     << "), None)";
  os << "\n";
  return success();
}

// Assuming StoreOp is only used while storing results.
// Example: temp_nodes[result_idx][idx] = temp_nodes[i]
// Note: memref::AllocOp and memref::StoreOp need to be in sync on how the
// indices are processed.
LogicalResult JaxiteWordEmitter::printOperation(memref::StoreOp op) {
  os << "temp_nodes[" << variableNames->getIntForValue(op.getMemref()) << "]";
  os << "["
     << flattenedIndex(
            op.getMemRefType(), op.getIndices(),
            [&](Value value) {
              return dyn_cast<IntegerAttr>(
                         dyn_cast<arith::ConstantOp>(value.getDefiningOp())
                             .getValue())
                  .getValue()
                  .getSExtValue();
            })
     << "]";
  os << " = " << "temp_nodes["
     << variableNames->getIntForValue(op.getValueToStore()) << "]";
  os << "\n";
  return success();
}

FailureOr<std::string> JaxiteWordEmitter::convertType(Type type) {
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
  return llvm::TypeSwitch<Type&, FailureOr<std::string>>(type)
      .Case<CiphertextType>(
          [&](auto type) { return std::string("jaxite_word.Ciphertext"); })
      .Case<ModulusListType>(
          [&](auto type) { return std::string("jaxite_word.ModulusList"); })
      .Default([&](Type&) { return failure(); });
}

LogicalResult JaxiteWordEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

JaxiteWordEmitter::JaxiteWordEmitter(raw_ostream& os,
                                     SelectVariableNames* variableNames)
    : os(os), variableNames(variableNames) {}

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir
