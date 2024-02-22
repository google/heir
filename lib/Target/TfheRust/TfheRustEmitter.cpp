#include "include/Target/TfheRust/TfheRustEmitter.h"

#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <string_view>

#include "include/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "include/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "include/Dialect/TfheRust/IR/TfheRustOps.h"
#include "include/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "lib/Target/TfheRust/TfheRustTemplates.h"
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
namespace tfhe_rust {

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

FailureOr<DenseElementsAttr> getConstantGlobalData(memref::GetGlobalOp op) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto globalOp =
      dyn_cast<mlir::memref::GlobalOp>(module.lookupSymbol(op.getName()));
  if (!globalOp) {
    return failure();
  }
  auto cstAttr =
      dyn_cast_or_null<DenseElementsAttr>(globalOp.getConstantInitValue());
  if (!cstAttr) {
    return failure();
  }
  return cstAttr;
}

}  // namespace

void registerToTfheRustTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-tfhe-rust",
      "translate the tfhe_rs dialect to Rust code for tfhe-rs",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToTfheRust(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, tfhe_rust::TfheRustDialect,
                        arith::ArithDialect, tensor::TensorDialect,
                        memref::MemRefDialect>();
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
          .Case<arith::ConstantOp, arith::ShRSIOp, arith::ShLIOp,
                arith::TruncIOp, arith::AndIOp>(
              [&](auto op) { return printOperation(op); })
          // TfheRust ops
          .Case<AddOp, ApplyLookupTableOp, BitAndOp, GenerateLookupTableOp,
                ScalarLeftShiftOp, CreateTrivialOp>(
              [&](auto op) { return printOperation(op); })
          // Tensor ops
          .Case<tensor::ExtractOp, tensor::FromElementsOp>(
              [&](auto op) { return printOperation(op); })
          // MemRef ops
          .Case<memref::GlobalOp, memref::DeallocOp>([&](auto op) {
            // These are no-ops.
            return success();
          })
          .Case<memref::AllocOp, memref::GetGlobalOp, memref::LoadOp,
                memref::StoreOp>(  // todo subview & copy
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
    os << argName << ": " << (isa<IntegerType>(arg.getType()) ? "" : "&");
    if (failed(emitType(arg.getType()))) {
      return funcOp.emitOpError()
             << "Failed to emit tfhe-rs type " << arg.getType();
    }
    os << ",\n";
    if (isa<tfhe_rust::ServerKeyType>(arg.getType())) {
      serverKeyArg_ = argName;
    }
  }
  os.unindent();
  os << ") -> ";

  if (serverKeyArg_.empty()) {
    return funcOp.emitWarning() << "expected server key function argument to "
                                   "create default ciphertexts";
  }

  if (funcOp.getNumResults() == 1) {
    Type result = funcOp.getResultTypes()[0];
    if (failed(emitType(result))) {
      return funcOp.emitOpError() << "Failed to emit tfhe-rs type " << result;
    }
  } else {
    auto result = commaSeparatedTypes(
        funcOp.getResultTypes(), [&](Type type) -> FailureOr<std::string> {
          auto result = convertType(type);
          if (failed(result)) {
            return funcOp.emitOpError()
                   << "Failed to emit tfhe-rs type " << type;
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

void TfheRustEmitter::emitAssignPrefix(Value result, bool mut,
                                       std::string type) {
  os << "let " << (mut ? "mut " : "") << variableNames->getNameForValue(result)
     << (type.empty() ? "" : (" : " + type)) << " = ";
}

LogicalResult TfheRustEmitter::printSksMethod(
    ::mlir::Value result, ::mlir::Value sks, ::mlir::ValueRange nonSksOperands,
    std::string_view op, SmallVector<std::string> operandTypes) {
  emitAssignPrefix(result);

  auto operandTypesIt = operandTypes.begin();
  os << variableNames->getNameForValue(sks) << "." << op << "(";
  os << commaSeparatedValues(nonSksOperands, [&](Value value) {
    const auto *prefix = value.getType().hasTrait<PassByReference>() ? "&" : "";
    return prefix + variableNames->getNameForValue(value) +
           (!operandTypes.empty() ? " as " + *operandTypesIt++ : "");
  });
  os << ");\n";
  return success();
}

LogicalResult TfheRustEmitter::printBinaryOp(::mlir::Value result,
                                             ::mlir::Value lhs,
                                             ::mlir::Value rhs,
                                             std::string_view op) {
  emitAssignPrefix(result);
  os << variableNames->getNameForValue(lhs) << " " << op << " "
     << variableNames->getNameForValue(rhs) << ";\n";
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

LogicalResult TfheRustEmitter::printOperation(GenerateLookupTableOp op) {
  auto sks = op.getServerKey();
  uint64_t truthTable = op.getTruthTable().getUInt();
  auto result = op.getResult();

  emitAssignPrefix(result);
  os << variableNames->getNameForValue(sks) << ".generate_lookup_table(";
  os << "|x| (" << std::to_string(truthTable) << " >> x) & 1";
  os << ");\n";
  return success();
}

LogicalResult TfheRustEmitter::printOperation(ScalarLeftShiftOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getCiphertext(), op.getShiftAmount()},
                        "scalar_left_shift");
}

LogicalResult TfheRustEmitter::printOperation(CreateTrivialOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(), {op.getValue()},
                        "create_trivial", {"u64"});
}

LogicalResult TfheRustEmitter::printOperation(arith::ConstantOp op) {
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

LogicalResult TfheRustEmitter::printOperation(::mlir::arith::ShLIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "<<");
}

LogicalResult TfheRustEmitter::printOperation(::mlir::arith::AndIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "&");
}

LogicalResult TfheRustEmitter::printOperation(::mlir::arith::ShRSIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), ">>");
}

LogicalResult TfheRustEmitter::printOperation(::mlir::arith::TruncIOp op) {
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

LogicalResult TfheRustEmitter::printOperation(tensor::ExtractOp op) {
  // We assume here that the indices are SSA values (not integer attributes).
  emitAssignPrefix(op.getResult());
  os << "&" << variableNames->getNameForValue(op.getTensor()) << "["
     << commaSeparatedValues(
            op.getIndices(),
            [&](Value value) { return variableNames->getNameForValue(value); })
     << "];\n";
  return success();
}

LogicalResult TfheRustEmitter::printOperation(tensor::FromElementsOp op) {
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

LogicalResult TfheRustEmitter::printOperation(memref::AllocOp op) {
  //  Uses an iterator to create an array with a default value
  MemRefType memRefType = op.getMemref().getType();
  auto typeStr = convertType(memRefType);
  if (failed(typeStr)) {
    op.emitOpError() << "failed to emit memref type " << memRefType;
  }
  emitAssignPrefix(op.getResult(), true, typeStr.value());

  auto defaultOr = defaultValue(memRefType.getElementType());
  if (failed(defaultOr)) {
    return op.emitOpError()
           << "Failed to emit default memref element type " << memRefType;
  }
  std::string res = defaultOr.value();
  for ([[maybe_unused]] unsigned dim : memRefType.getShape()) {
    res = llvm::formatv("core::array::from_fn(|_| {0})", res);
  }
  os << res << ";\n";
  return success();
}

LogicalResult TfheRustEmitter::printOperation(memref::GetGlobalOp op) {
  MemRefType memRefType = dyn_cast<MemRefType>(op.getResult().getType());
  if (!memRefType) {
    return op.emitOpError()
           << "Expected global to be a memref " << op.getName();
  }
  auto cstAttr = getConstantGlobalData(op);
  if (failed(cstAttr)) {
    return op.emitOpError() << "Failed to get constant global data";
  }

  auto type = convertType(memRefType.getElementType());
  if (failed(type)) {
    return op.emitOpError()
           << "Failed to emit type for global " << op.getResult().getType();
  }

  // Globals are emitted as 1-D arrays.
  os << "static " << variableNames->getNameForValue(op.getResult())
     << llvm::formatv(" : [{0}; {1}]", type, memRefType.getNumElements())
     << " = [";

  // Populate data by iterating through constant data attribute
  auto printValue = [](APInt value) -> std::string {
    llvm::SmallString<40> s;
    value.toStringSigned(s, 10);
    return std::string(s);
  };

  auto cstIter = cstAttr.value().value_begin<APInt>();
  auto cstIterEnd = cstAttr.value().value_end<APInt>();
  os << std::accumulate(
      std::next(cstIter), cstIterEnd, printValue(*cstIter),
      [&](std::string a, APInt value) { return a + ", " + printValue(value); });

  os << "];\n";
  return success();
}

LogicalResult TfheRustEmitter::printOperation(memref::StoreOp op) {
  os << variableNames->getNameForValue(op.getMemref()) << "["
     << commaSeparatedValues(
            op.getIndices(),
            [&](Value value) { return variableNames->getNameForValue(value); })
     << "] = " << variableNames->getNameForValue(op.getValueToStore()) << ";\n";
  return success();
}

LogicalResult TfheRustEmitter::printOperation(memref::LoadOp op) {
  emitAssignPrefix(op.getResult());
  bool isRef = isa<EncryptedUInt3Type>(op.getResult().getType());
  os << (isRef ? "&" : "") << variableNames->getNameForValue(op.getMemref())
     << "[";

  if (dyn_cast_or_null<memref::GetGlobalOp>(op.getMemRef().getDefiningOp())) {
    // Global arrays are 1-dimensional, so flatten the index.
    // TODO(#449): Share with Verilog Emitter.
    const auto [strides, offset] =
        getStridesAndOffset(cast<MemRefType>(op.getMemRefType()));
    os << std::to_string(offset);
    for (int i = 0; i < strides.size(); ++i) {
      os << llvm::formatv(" + {0} * {1}",
                          variableNames->getNameForValue(op.getIndices()[i]),
                          strides[i]);
    }
  } else {
    os << commaSeparatedValues(op.getIndices(), [&](Value value) {
      return variableNames->getNameForValue(value);
    });
  }

  os << "];\n";
  return success();
}

FailureOr<std::string> TfheRustEmitter::convertType(Type type) {
  // Note: these are probably not the right type names to use exactly, and they
  // will need to chance to the right values once we try to compile it against
  // a specific API version.
  return llvm::TypeSwitch<Type &, FailureOr<std::string>>(type)
      .Case<RankedTensorType>(
          [&](RankedTensorType type) -> FailureOr<std::string> {
            // Tensor types are emitted as vectors
            auto elementTy = convertType(type.getElementType());
            if (failed(elementTy)) return failure();
            return std::string("Vec<" + elementTy.value() + ">");
          })
      .Case<MemRefType>([&](MemRefType type) -> FailureOr<std::string> {
        // MemRef types are emitted as arrays
        auto elementTy = convertType(type.getElementType());
        if (failed(elementTy)) return failure();
        std::string res = elementTy.value();
        for (unsigned dim : type.getShape()) {
          res = llvm::formatv("[{0}; {1}]", res, dim);
        }
        return res;
      })
      .Case<IntegerType>([&](IntegerType type) -> FailureOr<std::string> {
        if (type.getWidth() == 1) {
          return std::string("bool");
        }
        auto width = getRustIntegerType(type.getWidth());
        if (failed(width)) return failure();
        return (type.isUnsigned() ? std::string("u") : "") + "i" +
               std::to_string(width.value());
      })
      .Case<EncryptedUInt3Type>(
          [&](auto type) { return std::string("Ciphertext"); })
      .Case<ServerKeyType>([&](auto type) { return std::string("ServerKey"); })
      .Case<LookupTableType>(
          [&](auto type) { return std::string("LookupTableOwned"); })
      .Default([&](Type &) { return failure(); });
}

FailureOr<std::string> TfheRustEmitter::defaultValue(Type type) {
  return llvm::TypeSwitch<Type &, FailureOr<std::string>>(type)
      .Case<IntegerType>([&](IntegerType type) { return std::string("0"); })
      .Case<EncryptedUInt3Type>([&](auto type) -> FailureOr<std::string> {
        if (serverKeyArg_.empty()) return failure();
        return std::string(
            llvm::formatv("{0}.create_trivial(0 as u64)", serverKeyArg_));
      })
      .Default([&](Type &) { return failure(); });
}

LogicalResult TfheRustEmitter::printOperation(BitAndOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getLhs(), op.getRhs()}, "bitand");
}

LogicalResult TfheRustEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

TfheRustEmitter::TfheRustEmitter(raw_ostream &os,
                                 SelectVariableNames *variableNames)
    : os(os), variableNames(variableNames) {}
}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir
