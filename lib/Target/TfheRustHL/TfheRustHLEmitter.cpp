#include "lib/Target/TfheRustHL/TfheRustHLEmitter.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "lib/Target/TfheRust/Utils.h"
#include "lib/Target/TfheRustHL/TfheRustHLTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
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

#define DEBUG_TYPE "emit-tfhe-rust-hl"

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

// Global Variable
// Here the input size of the function arguments is stored
int16_t DefaultTfheRustHLBitWidth = 32;

void registerToTfheRustHLTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-tfhe-rust-hl", "translate the tfhe-rs dialect to HL Rust code",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToTfheRustHL(op, output);
      },
      [](DialectRegistry& registry) {
        registry.insert<func::FuncDialect, tfhe_rust::TfheRustDialect,
                        affine::AffineDialect, arith::ArithDialect,
                        tensor::TensorDialect, memref::MemRefDialect>();
      });
}

LogicalResult translateToTfheRustHL(Operation* op, llvm::raw_ostream& os) {
  SelectVariableNames variableNames(op);
  TfheRustHLEmitter emitter(os, &variableNames);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult TfheRustHLEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::CallOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Affine ops
          .Case<affine::AffineForOp, affine::AffineYieldOp,
                affine::AffineLoadOp, affine::AffineStoreOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp, arith::IndexCastOp, arith::ShRSIOp,
                arith::ShLIOp, arith::TruncIOp, arith::AndIOp>(
              [&](auto op) { return printOperation(op); })
          // MemRef ops
          .Case<memref::AllocOp, memref::DeallocOp, memref::LoadOp,
                memref::GetGlobalOp, memref::StoreOp>(
              [&](auto op) { return printOperation(op); })
          // MemRef ops
          .Case<memref::GlobalOp, memref::DeallocOp>([&](auto op) {
            // These are no-ops.
            return success();
          })
          // TfheRust ops
          .Case<AddOp, SubOp, MulOp, ScalarRightShiftOp, CastOp,
                CreateTrivialOp, BitAndOp, BitOrOp, BitXorOp>(
              [&](auto op) { return printOperation(op); })
          // Tensor ops
          .Case<tensor::ExtractOp, tensor::FromElementsOp, tensor::InsertOp>(
              [&](auto op) { return printOperation(op); })
          .Default([&](Operation&) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";

  // Find default type of the module and use a Type alias
  moduleOp.getOperation()->walk([&](Operation* op) {
    for (Type resultType : op->getResultTypes()) {
      if (resultType.hasTrait<EncryptedInteger>()) {
        auto size = getTfheRustBitWidth(resultType);
        DefaultTfheRustHLBitWidth = size;

        os << "type Ciphertext = tfhe::FheUint<tfhe::FheUint" << size
           << "Id>; \n\n";

        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();  // Continue walking to the next operation
  });

  // Emit rest the actual program
  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(func::FuncOp funcOp) {
  if (failed(tfhe_rust::canEmitFuncForTfheRust(funcOp))) {
    // Return success implies print nothing, and note the called function
    // emits a warning.
    return success();
  }

  os << "pub fn " << funcOp.getName() << "(\n";
  os.indent();
  for (Value arg : funcOp.getArguments()) {
    if (!isa<tfhe_rust::ServerKeyType>(arg.getType())) {
      auto argName = variableNames->getNameForValue(arg);
      os << argName << ": &";
      if (failed(emitType(arg.getType()))) {
        return funcOp.emitOpError()
               << "Failed to emit tfhe-rs type " << arg.getType();
      }
      os << ",\n";
    }
  }
  os.unindent();
  os << ") -> ";

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

  for (Block& block : funcOp.getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }

  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(func::ReturnOp op) {
  std::function<std::string(Value)> valueOrClonedValue = [&](Value value) {
    if (isa<BlockArgument>(value)) {
      // Function arguments used as outputs must be cloned.
      return variableNames->getNameForValue(value) + ".clone()";
    } else if (MemRefType memRefType = dyn_cast<MemRefType>(value.getType())) {
      auto shape = memRefType.getShape();
      // Internally allocated memrefs that are treated as hashmaps must be
      // converted to arrays.
      unsigned int i = 0;
      std::string res =
          variableNames->getNameForValue(value) + std::string(".get(&(") +
          std::accumulate(std::next(shape.begin()), shape.end(),
                          std::string("i0"),
                          [&](const std::string& a, int64_t value) {
                            return a + ", i" + std::to_string(++i);
                          }) +
          std::string(")).unwrap().clone()");
      for ([[maybe_unused]] unsigned _ : shape) {
        res = llvm::formatv("core::array::from_fn(|i{0}| {1})", i--, res);
      }
      return res;
    }
    return variableNames->getNameForValue(value);
  };

  if (op.getNumOperands() == 0) {
    return success();
  }

  if (op.getNumOperands() == 1) {
    os << valueOrClonedValue(op.getOperands()[0]) << "\n";
    return success();
  }

  os << "(" << commaSeparatedValues(op.getOperands(), valueOrClonedValue)
     << ")\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(func::CallOp op) {
  os << "let " << variableNames->getNameForValue(op->getResult(0)) << " = ";

  os << op.getCallee() << "(";
  for (Value arg : op->getOperands()) {
    if (!isa<tfhe_rust::ServerKeyType>(arg.getType())) {
      auto argName = variableNames->getNameForValue(arg);
      if (op.getOperands().back() == arg) {
        os << "&" << argName;
      } else {
        os << "&" << argName << ", ";
      }
    }
  }

  os << "); \n";
  return success();
}

void TfheRustHLEmitter::emitAssignPrefix(Value result) {
  os << "let " << variableNames->getNameForValue(result) << " = ";
}

LogicalResult TfheRustHLEmitter::printMethod(
    ::mlir::Value result, ::mlir::ValueRange nonSksOperands,
    std::string_view op, SmallVector<std::string> operandTypes) {
  emitAssignPrefix(result);

  auto* operandTypesIt = operandTypes.begin();
  os << commaSeparatedValues(nonSksOperands, [&](Value value) {
    auto* prefix = value.getType().hasTrait<PassByReference>() ? "&" : "";
    // First check if a DefiningOp exists
    // if not: comes from function definition
    mlir::Operation* op = value.getDefiningOp();
    if (op) {
      auto referencePredicate =
          isa<tensor::ExtractOp>(op) || isa<memref::LoadOp>(op);
      prefix = referencePredicate ? "" : prefix;
    } else {
      prefix = "";
    }

    return prefix + variableNames->getNameForValue(value) +
           (!operandTypes.empty() ? " as " + *operandTypesIt++ : "");
  });

  os << ");\n";

  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(CreateTrivialOp op) {
  emitAssignPrefix(op.getResult());

  os << "FheUint" << getTfheRustBitWidth(op.getResult().getType())
     << "::try_encrypt_trivial("
     << variableNames->getNameForValue(op.getValue());

  if (op.getValue().getType().isSigned())
    os << " as u" << getTfheRustBitWidth(op.getResult().getType());

  os << ").unwrap();\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(affine::AffineForOp op) {
  if (op.getStepAsInt() > 1) {
    return op.emitOpError() << "AffineForOp has step > 1";
  }
  os << "for " << variableNames->getNameForValue(op.getInductionVar()) << " in "
     << op.getConstantLowerBound() << ".." << op.getConstantUpperBound()
     << " {\n";
  os.indent();

  // Walk the body of the parallel operation in Program order
  for (auto& op : op.getBody()->getOperations()) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(affine::AffineYieldOp op) {
  if (op->getNumResults() != 0) {
    return op.emitOpError() << "AffineYieldOp has non-zero number of results";
  }
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  if (isa<IntegerType>(op.getType()) &&
      op.getType().getIntOrFloatBitWidth() == 1) {
    os << "let " << variableNames->getNameForValue(op.getResult())
       << " : bool = ";
    os << (cast<IntegerAttr>(valueAttr).getValue().isZero() ? "false" : "true")
       << ";\n";
    return success();
  }

  // TODO(#1303): Add signed integer support to HL Emitter
  // By default, it emits an unsigned integer.
  emitAssignPrefix(op.getResult());
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    os << intAttr.getValue().abs() << convertType(op.getType()) << ";\n";
  } else {
    return op.emitError() << "Unknown constant type " << valueAttr.getType();
  }
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(arith::IndexCastOp op) {
  emitAssignPrefix(op.getOut());
  os << variableNames->getNameForValue(op.getIn()) << " as ";
  if (failed(emitType(op.getOut().getType()))) {
    return op.emitOpError()
           << "Failed to emit index cast type " << op.getOut().getType();
  }
  os << ";\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printBinaryOp(::mlir::Value result,
                                               ::mlir::Value lhs,
                                               ::mlir::Value rhs,
                                               std::string_view op) {
  emitAssignPrefix(result);

  if (auto cteOp =
          dyn_cast_or_null<mlir::arith::ConstantOp>(rhs.getDefiningOp())) {
    auto intValue =
        cast<IntegerAttr>(cteOp.getValue()).getValue().getZExtValue();
    os << checkOrigin(lhs) << variableNames->getNameForValue(lhs) << " " << op
       << " " << intValue << "u" << cteOp.getType().getIntOrFloatBitWidth()
       << ";\n";
    return success();
  }

  // Note: arith.constant op requires signless integer types, but here we
  // manually emit an unsigned integer type.
  os << checkOrigin(lhs) << variableNames->getNameForValue(lhs) << " " << op
     << " " << checkOrigin(rhs) << variableNames->getNameForValue(rhs) << ";\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(::mlir::arith::ShLIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "<<");
}

LogicalResult TfheRustHLEmitter::printOperation(::mlir::arith::AndIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "&");
}

LogicalResult TfheRustHLEmitter::printOperation(::mlir::arith::ShRSIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), ">>");
}

LogicalResult TfheRustHLEmitter::printOperation(::mlir::arith::TruncIOp op) {
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
LogicalResult TfheRustHLEmitter::printOperation(memref::AllocOp op) {
  os << "let mut " << variableNames->getNameForValue(op.getMemref())
     << " : BTreeMap<("
     << std::accumulate(
            std::next(op.getMemref().getType().getShape().begin()),
            op.getMemref().getType().getShape().end(), std::string("usize"),
            [&](const std::string& a, int64_t value) { return a + ", usize"; })
     << "), ";
  if (failed(emitType(op.getMemref().getType().getElementType()))) {
    return op.emitOpError() << "Failed to get memref element type";
  }
  os << "> = BTreeMap::new();\n";

  return success();
}

// Use a BTreeMap<(usize, ...), Ciphertext>.
LogicalResult TfheRustHLEmitter::printOperation(memref::DeallocOp op) {
  os << variableNames->getNameForValue(op.getMemref()) << ".clear();\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(memref::GetGlobalOp op) {
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
  auto printValue = [](const APInt& value) -> std::string {
    llvm::SmallString<40> s;
    value.toStringSigned(s, 10);
    return std::string(s);
  };

  auto cstIter = cstAttr.value().value_begin<APInt>();
  auto cstIterEnd = cstAttr.value().value_end<APInt>();
  os << std::accumulate(std::next(cstIter), cstIterEnd, printValue(*cstIter),
                        [&](const std::string& a, const APInt& value) {
                          return a + ", " + printValue(value);
                        });

  os << "];\n";
  return success();
}

// Store into a BTreeMap<(usize, ...), Ciphertext>
LogicalResult TfheRustHLEmitter::printOperation(memref::StoreOp op) {
  // We assume here that the indices are SSA values (not integer attributes).
  os << variableNames->getNameForValue(op.getMemref());
  os << ".insert((" << commaSeparatedValues(op.getIndices(), [&](Value value) {
    return variableNames->getNameForValue(value) + std::string(" as usize");
  }) << "), ";

  // Note: we may not need to clone all the time, but the BTreeMap stores
  // Ciphertexts, not &Ciphertexts. This is because results computed inside for
  // loops will not live long enough.
  const auto* suffix = ".clone()";
  os << variableNames->getNameForValue(op.getValueToStore()) << suffix
     << ");\n";
  return success();
}

// Produces a &Ciphertext
LogicalResult TfheRustHLEmitter::printOperation(memref::LoadOp op) {
  emitAssignPrefix(op.getResult());

  // We assume here that the indices are SSA values (not integer attributes).
  if (dyn_cast_or_null<memref::GetGlobalOp>(op.getMemRef().getDefiningOp())) {
    // Global arrays are 1-dimensional, so flatten the index
    os << variableNames->getNameForValue(op.getMemref());

    os << "["
       << flattenIndexExpressionSOP(
              op.getMemRefType(), op.getIndices(), [&](Value value) {
                return variableNames->getNameForValue(value);
              });
    os << "]; \n";
    return success();
  }

  if (isa<BlockArgument>(op.getMemref())) {
    os << "&" << variableNames->getNameForValue(op.getMemRef());
    for (auto value : op.getIndices()) {
      os << "[" << variableNames->getNameForValue(value) << "]";
    }
    os << ";\n";
    return success();
  }

  // Treat this as a BTreeMap
  os << variableNames->getNameForValue(op.getMemref());

  os << ".get(&(" << commaSeparatedValues(op.getIndices(), [&](Value value) {
    return variableNames->getNameForValue(value) + " as usize";
  }) << ")).unwrap();\n";

  return success();
}

// FIXME?: This is a hack to get the index of the value
static int extractIntFromValue(Value value) {
  if (auto ctOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp())) {
    return cast<IntegerAttr>(ctOp.getValue()).getValue().getSExtValue();
  }
  return -1;
}

// Store into a BTreeMap<(usize, ...), Ciphertext>
LogicalResult TfheRustHLEmitter::printOperation(affine::AffineStoreOp op) {
  // We assume here that the indices are SSA values (not integer attributes).

  OpBuilder builder(op->getContext());
  auto indices = affine::expandAffineMap(builder, op->getLoc(), op.getMap(),
                                         op.getIndices());

  os << variableNames->getNameForValue(op.getMemref());
  os << ".insert((" << commaSeparatedValues(indices.value(), [&](Value value) {
    return std::to_string(extractIntFromValue(value)) +
           std::string(" as usize");
  }) << "), ";

  // Note: we may not need to clone all the time, but the BTreeMap stores
  // Ciphertexts, not &Ciphertexts. This is because results computed inside for
  // loops will not live long enough.

  const auto* suffix = ".clone()";
  os << variableNames->getNameForValue(op.getValueToStore()) << suffix
     << ");\n";
  return success();
}

// Produces a &Ciphertext
LogicalResult TfheRustHLEmitter::printOperation(affine::AffineLoadOp op) {
  OpBuilder builder(op->getContext());
  auto indices = affine::expandAffineMap(builder, op->getLoc(), op.getMap(),
                                         op.getIndices());

  if (isa<BlockArgument>(op.getMemref())) {
    emitAssignPrefix(op.getResult());

    os << "&" << variableNames->getNameForValue(op.getMemRef());
    for (auto value : indices.value()) {
      os << "[" << std::to_string(extractIntFromValue(value)) << "]";
    }
    os << ";\n";
    return success();
  }

  // Treat this as a BTreeMap
  emitAssignPrefix(op.getResult());
  os << "&" << variableNames->getNameForValue(op.getMemref()) << ".get(&("
     << commaSeparatedValues(
            indices.value(),
            [&](Value value) {
              return std::to_string(extractIntFromValue(value));
            })
     << ")).unwrap();\n";
  return success();
}

// Produces a &Ciphertext
LogicalResult TfheRustHLEmitter::printOperation(tensor::ExtractOp op) {
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
LogicalResult TfheRustHLEmitter::printOperation(tensor::FromElementsOp op) {
  emitAssignPrefix(op.getResult());
  os << "vec![" << commaSeparatedValues(op.getOperands(), [&](Value value) {
    // Check if block argument, if so, clone.
    const auto* cloneStr = isa<BlockArgument>(value) ? ".clone()" : "";
    // Get the name of defining operation its dialect
    auto tfheOp =
        value.getDefiningOp()->getDialect()->getNamespace() == "tfhe_rust";
    const auto* prefix = tfheOp ? "&" : "";
    return std::string(prefix) + variableNames->getNameForValue(value) +
           cloneStr;
  }) << "];\n";
  return success();
}

// Does not need to produce a value
LogicalResult TfheRustHLEmitter::printOperation(tensor::InsertOp op) {
  emitAssignPrefix(op.getResult());
  os << "vec![" << commaSeparatedValues(op.getOperands(), [&](Value value) {
    // Check if block argument, if so, clone.
    const auto* cloneStr = isa<BlockArgument>(value) ? ".clone()" : "";
    // Get the name of defining operation its dialect
    auto tfheOp =
        value.getDefiningOp()->getDialect()->getNamespace() == "tfhe_rust";
    const auto* prefix = tfheOp ? "&" : "";
    return std::string(prefix) + variableNames->getNameForValue(value) +
           cloneStr;
  }) << "];\n";

  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(BitAndOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "&");
}

LogicalResult TfheRustHLEmitter::printOperation(BitOrOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "|");
}

LogicalResult TfheRustHLEmitter::printOperation(BitXorOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "^");
}

LogicalResult TfheRustHLEmitter::printOperation(AddOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "+");
}

LogicalResult TfheRustHLEmitter::printOperation(MulOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "*");
}

LogicalResult TfheRustHLEmitter::printOperation(SubOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "-");
}

LogicalResult TfheRustHLEmitter::printOperation(ScalarRightShiftOp op) {
  emitAssignPrefix(op.getResult());

  os << checkOrigin(op.getCiphertext())
     << variableNames->getNameForValue(op.getCiphertext()) << " >> "
     << op.getShiftAmount() << "u8;\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(CastOp op) {
  auto resultTypeSize = getTfheRustBitWidth(op.getOutput().getType());

  emitAssignPrefix(op.getResult());
  os << "FheUint" << resultTypeSize << "::cast_from(";
  os << variableNames->getNameForValue(op.getCiphertext()) << ".clone());\n";

  return success();
}

FailureOr<std::string> TfheRustHLEmitter::convertType(Type type) {
  // Note: these are probably not the right type names to use exactly, and
  // they will need to chance to the right values once we try to compile it
  // against a specific API version.

  if (type.hasTrait<EncryptedInteger>()) {
    auto ctxtWidth = getTfheRustBitWidth(type);
    if (ctxtWidth == DefaultTfheRustHLBitWidth) {
      return std::string("Ciphertext");
    }
    return "tfhe::FheUint<tfhe::FheUint" + std::to_string(ctxtWidth) + "Id>";
    ;
  }

  return llvm::TypeSwitch<Type&, FailureOr<std::string>>(type)
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
        for (unsigned dim : llvm::reverse(type.getShape())) {
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
        return (type.isSigned() ? std::string("i") : std::string("u")) +
               std::to_string(width.value());
      })
      .Case<IndexType>([&](IndexType type) -> FailureOr<std::string> {
        return std::string("usize");
      })
      .Case<LookupTableType>(
          [&](auto type) { return std::string("LookupTableOwned"); })
      .Default([&](Type&) { return failure(); });
}

LogicalResult TfheRustHLEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

std::string TfheRustHLEmitter::checkOrigin(Value value) {
  // First check if a DefiningOp exists
  // if not: comes from function definition
  mlir::Operation* opParent = value.getDefiningOp();
  if (opParent) {
    if (!isa<tensor::FromElementsOp, tensor::ExtractOp, affine::AffineLoadOp,
             memref::LoadOp>(opParent))
      return "&";

  } else {
    return "";
  }

  return "";
}

TfheRustHLEmitter::TfheRustHLEmitter(raw_ostream& os,
                                     SelectVariableNames* variableNames)
    : os(os), variableNames(variableNames) {}
}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir
