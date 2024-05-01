#include "lib/Target/TfheRust/TfheRustEmitter.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "lib/Graph/Graph.h"
#include "lib/Target/TfheRust/TfheRustTemplates.h"
#include "lib/Target/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/CommandLine.h"     // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"   // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

#define DEBUG_TYPE "tfhe-rust-emitter"

namespace mlir {
namespace heir {
namespace tfhe_rust {

namespace {

bool isLevelledOp(Operation *op) {
  return isa<ApplyLookupTableOp, AddOp, ScalarLeftShiftOp>(op);
}

bool usedByLevelledOp(Value value) {
  return llvm::any_of(value.getUsers(),
                      [](Operation *op) { return isLevelledOp(op); });
}

bool usedByNonLevelledOp(Value value) {
  return llvm::any_of(value.getUsers(),
                      [](Operation *op) { return !isLevelledOp(op); });
}

std::pair<graph::Graph<Operation *>, Operation *> getGraph(Operation *op) {
  graph::Graph<Operation *> graph;

  auto block = op->getBlock();
  while (op != nullptr) {
    if (!isLevelledOp(op)) {
      return {graph, op};
    }
    graph.addVertex(op);
    for (auto operand : op->getOperands()) {
      auto *definingOp = operand.getDefiningOp();
      if (!definingOp || definingOp->getBlock() != block ||
          !isLevelledOp(definingOp)) {
        continue;
      }
      graph.addEdge(definingOp, op);
    }
    op = op->getNextNode();
  }

  return {graph, op};
}

SmallVector<Value> getCiphertextOperands(ValueRange inputs) {
  SmallVector<Value> vals;
  for (Value val : inputs) {
    // TODO(#474): Generalize to any encrypted uint.
    if (isa<tfhe_rust::EncryptedUInt3Type>(val.getType())) {
      vals.push_back(val);
    }
  }

  return vals;
}

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

bool useLevels;

static llvm::cl::opt<bool, true> useLevelsFlag("use-levels",
                                               llvm::cl::desc("Use levels"),
                                               llvm::cl::location(useLevels),
                                               llvm::cl::init(false));

void registerToTfheRustTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-tfhe-rust",
      "translate the tfhe_rs dialect to Rust code for tfhe-rs",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToTfheRust(op, output, useLevels);
      },
      [](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, tfhe_rust::TfheRustDialect,
                        arith::ArithDialect, tensor::TensorDialect,
                        memref::MemRefDialect, affine::AffineDialect>();
      });
}

LogicalResult translateToTfheRust(Operation *op, llvm::raw_ostream &os,
                                  bool useLevels) {
  SelectVariableNames variableNames(op);
  TfheRustEmitter emitter(os, &variableNames, useLevels);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult TfheRustEmitter::emitBlock(::mlir::Operation *op) {
  // Translate ops in the block until we get to a tfhe_rust.ApplyLookupTable,
  // AddOp, ScalarLeftShitOp
  while (op != nullptr && !isLevelledOp(op)) {
    if (failed(translate(*op))) {
      return failure();
    }
    op = op->getNextNode();
  }
  if (op == nullptr) {
    return success();
  }
  // Compute a graph of the levelled operations.
  auto [graph, nextOp] = getGraph(op);
  if (!graph.empty()) {
    auto sortedGraph = graph.sortGraphByLevels();
    if (failed(sortedGraph)) {
      llvm_unreachable("Only possible failure is a cycle in the SSA graph!");
    }
    auto levels = sortedGraph.value();
    // Print lists of operations per level.
    for (int level = 0; level < levels.size(); ++level) {
      os << "static LEVEL_" << level << " : [((OpType, usize), &[GateInput]); "
         << levels[level].size() << "] = [";
      for (auto &op : levels[level]) {
        // Print the operation type and its ciphertext args
        os << llvm::formatv(
            "(({0}, {1}), &[{2}]), ", operationType(op),
            variableNames->getIntForValue(op->getResult(0)),
            commaSeparatedValues(
                getCiphertextOperands(op->getOperands()), [&](Value value) {
                  // TODO(#462): This assumes that all ciphertexts are
                  // loaded into temp_nodes. Currently, block arguments are
                  // not supported.
                  return "Tv(" +
                         std::to_string(variableNames->getIntForValue(value)) +
                         ")";
                }));
      }
      os << "];\n";
    }

    // Execute each task in the level.
    for (int level = 0; level < levels.size(); ++level) {
      os << llvm::formatv(
          "run_level({1}, &mut temp_nodes, &mut luts, &LEVEL_{0});\n", level,
          serverKeyArg_);
    }
  }
  // Continue to emit the block.
  return emitBlock(nextOp);
}

LogicalResult TfheRustEmitter::translateBlock(Block &block) {
  if (useLevels_) {
    Operation *op = &block.getOperations().front();
    return emitBlock(op);
  }
  for (Operation &op : block.getOperations()) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult TfheRustEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Affine ops
          .Case<affine::AffineForOp>(
              [&](auto op) { return printOperation(op); })
          .Case<affine::AffineYieldOp>([&](auto op) -> LogicalResult {
            if (op->getNumResults() != 0) {
              return op.emitOpError()
                     << "AffineYieldOp has non-zero number of results";
            }
            return success();
          })
          // Arith ops
          .Case<arith::ConstantOp, arith::IndexCastOp, arith::ShRSIOp,
                arith::ShLIOp, arith::TruncIOp, arith::AndIOp>(
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
                memref::StoreOp>([&](auto op) { return printOperation(op); })
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
  os << ")";

  if (serverKeyArg_.empty()) {
    return funcOp.emitWarning() << "expected server key function argument to "
                                   "create default ciphertexts";
  }

  if (funcOp.getNumResults() > 0) {
    os << " -> ";
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
  }

  os << " {\n";
  os.indent();

  // Create a global temp_nodes hashmap for any created SSA values.
  // TODO(#462): Insert block argument that are encrypted ints into
  // temp_nodes.
  if (useLevels_) {
    os << "let mut temp_nodes : HashMap<usize, Ciphertext> = "
          "HashMap::new();\n";
    os << "let mut luts : HashMap<&str, LookupTableOwned> = "
          "HashMap::new();\n";
    os << kRunLevelDefn << "\n";
  }

  for (Block &block : funcOp.getBlocks()) {
    if (failed(translateBlock(block))) {
      return funcOp.emitOpError()
             << "Failed to translate block of func " << funcOp.getName();
    }
  }

  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult TfheRustEmitter::printOperation(func::ReturnOp op) {
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
                          [&](std::string a, int64_t value) {
                            return a + ", i" + std::to_string(++i);
                          }) +
          std::string(")).unwrap().clone()");
      for ([[maybe_unused]] unsigned _ : shape) {
        res = llvm::formatv("core::array::from_fn(|i{0}| {1})", i--, res);
      }
      return res;
    } else if (isLevelledOp(value.getDefiningOp()) && useLevels_) {
      // This is from a levelled op stored in temp nodes.
      return std::string(
          llvm::formatv("temp_nodes[&{0}]",
                        std::to_string(variableNames->getIntForValue(value))));
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

void TfheRustEmitter::emitAssignPrefix(Value result, bool mut,
                                       std::string type) {
  os << "let " << (mut ? "mut " : "") << variableNames->getNameForValue(result)
     << (type.empty() ? "" : (" : " + type)) << " = ";
}

LogicalResult TfheRustEmitter::printSksMethod(
    ::mlir::Value result, ::mlir::Value sks, ::mlir::ValueRange nonSksOperands,
    std::string_view op, SmallVector<std::string> operandTypes) {
  emitAssignPrefix(result);

  if (!operandTypes.empty()) {
    assert(operandTypes.size() == nonSksOperands.size() &&
           "invalid sizes of operandTypes");
    operandTypes =
        llvm::to_vector(llvm::map_range(operandTypes, [&](std::string value) {
          return value.empty() ? "" : " as " + value;
        }));
  }
  auto *operandTypesIt = operandTypes.begin();
  os << variableNames->getNameForValue(sks) << "." << op << "(";
  os << commaSeparatedValues(nonSksOperands, [&](Value value) {
    auto valueStr = variableNames->getNameForValue(value);
    if (isa<LookupTableType>(value.getType()) && useLevels_) {
      valueStr = "luts[\"" + variableNames->getNameForValue(value) + "\"]";
    }
    std::string prefix = value.getType().hasTrait<PassByReference>() ? "&" : "";
    std::string suffix = operandTypes.empty() ? "" : *operandTypesIt++;
    return prefix + valueStr + suffix;
  });

  os << ");\n";

  // Insert ciphertext results into temp_nodes so that the levelled ops can
  // reference them.
  if (usedByLevelledOp(result) && useLevels_) {
    os << llvm::formatv("temp_nodes.insert({0}, {1}.clone());\n",
                        variableNames->getIntForValue(result),
                        variableNames->getNameForValue(result));
  }
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

  if (useLevels_) {
    os << "luts.insert(\"" << variableNames->getNameForValue(result) << "\", ";
  } else {
    emitAssignPrefix(result);
  }
  os << variableNames->getNameForValue(sks) << ".generate_lookup_table(";
  os << "|x| (" << std::to_string(truthTable) << " >> x) & 1)";

  if (useLevels_) {
    os << ")";
  }
  os << ";\n";
  return success();
}

std::string TfheRustEmitter::operationType(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::string>(op)
      .Case<tfhe_rust::ApplyLookupTableOp>([&](ApplyLookupTableOp op) {
        return "LUT3(\"" + variableNames->getNameForValue(op.getLookupTable()) +
               "\")";
      })
      .Case<tfhe_rust::ScalarLeftShiftOp>([&](ScalarLeftShiftOp op) {
        auto constantShift =
            cast<arith::ConstantOp>(op.getShiftAmount().getDefiningOp());
        return "LSH(" +
               std::to_string(
                   cast<IntegerAttr>(constantShift.getValue()).getInt()) +
               ")";
      })
      .Case<tfhe_rust::AddOp>([&](Operation *) { return "ADD"; });
}

LogicalResult TfheRustEmitter::printOperation(affine::AffineForOp forOp) {
  os << "for " << variableNames->getNameForValue(forOp.getInductionVar())
     << " in " << forOp.getConstantLowerBound() << ".."
     << forOp.getConstantUpperBound() << " {\n";
  os.indent();

  if (failed(translateBlock(*forOp.getBody()))) {
    return forOp.emitOpError() << "Failed to translate for loop block";
  }

  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult TfheRustEmitter::printOperation(ScalarLeftShiftOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(),
                        {op.getCiphertext(), op.getShiftAmount()},
                        "scalar_left_shift", {"", "u8"});
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

LogicalResult TfheRustEmitter::printOperation(arith::IndexCastOp op) {
  emitAssignPrefix(op.getOut());
  os << variableNames->getNameForValue(op.getIn()) << " as ";
  if (failed(emitType(op.getOut().getType()))) {
    return op.emitOpError()
           << "Failed to emit index cast type " << op.getOut().getType();
  }
  os << ";\n";
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
  // We assume here that the indices are SSA values (not integer
  // attributes).
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
  os << "let mut " << variableNames->getNameForValue(op.getMemref())
     << " : HashMap<("
     << std::accumulate(
            std::next(op.getMemref().getType().getShape().begin()),
            op.getMemref().getType().getShape().end(), std::string("usize"),
            [&](std::string a, int64_t value) { return a + ", usize"; })
     << "), ";
  if (failed(emitType(op.getMemref().getType().getElementType()))) {
    return op.emitOpError() << "Failed to get memref element type";
  }

  os << "> = HashMap::new();\n";
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

void TfheRustEmitter::printStoreOp(memref::StoreOp op,
                                   std::string valueToStore) {
  os << variableNames->getNameForValue(op.getMemref());
  os << ".insert(("
     << commaSeparatedValues(op.getIndices(),
                             [&](Value value) {
                               return variableNames->getNameForValue(value) +
                                      std::string(" as usize");
                             })
     << "), " << valueToStore << ");\n";
}

LogicalResult TfheRustEmitter::printOperation(memref::StoreOp op) {
  auto valueToStore = variableNames->getNameForValue(op.getValueToStore());

  if (isLevelledOp(op.getValueToStore().getDefiningOp()) && useLevels_) {
    valueToStore =
        llvm::formatv("temp_nodes[&{0}].clone()",
                      variableNames->getIntForValue(op.getValueToStore()));
  } else if (isa<tfhe_rust::TfheRustDialect>(
                 op.getValueToStore().getType().getDialect()) &&
             !isa<tfhe_rust::TfheRustDialect>(
                 op.getValueToStore().getDefiningOp()->getDialect())) {
    valueToStore += ".clone()";
  }

  printStoreOp(op, valueToStore);
  return success();
}

void TfheRustEmitter::printLoadOp(memref::LoadOp op) {
  os << variableNames->getNameForValue(op.getMemref());
  if (dyn_cast_or_null<memref::GetGlobalOp>(op.getMemRef().getDefiningOp())) {
    // Global arrays are 1-dimensional, so flatten the index
    // TODO(#449): Share with Verilog Emitter.
    const auto [strides, offset] =
        getStridesAndOffset(cast<MemRefType>(op.getMemRefType()));
    os << "[" << std::to_string(offset);
    for (int i = 0; i < strides.size(); ++i) {
      os << llvm::formatv(" + {0} * {1}",
                          variableNames->getNameForValue(op.getIndices()[i]),
                          strides[i]);
    }
    os << "]";
  } else if (isa<BlockArgument>(op.getMemRef())) {
    // This is a block argument array.
    os << bracketEnclosedValues(op.getIndices(), [&](Value value) {
      return variableNames->getNameForValue(value);
    });
  } else {
    // Otherwise, this must be an internally allocated memref, treated as a
    // hashmap.
    os << ".get(&(" << commaSeparatedValues(op.getIndices(), [&](Value value) {
      return variableNames->getNameForValue(value) + " as usize";
    }) << ")).unwrap()";
  }
}

LogicalResult TfheRustEmitter::printOperation(memref::LoadOp op) {
  // If the load op result is used in a levelled op, insert it into the
  // temp_nodes map.
  if (usedByLevelledOp(op) && useLevels_) {
    os << llvm::formatv("temp_nodes.insert({0}, ",
                        variableNames->getIntForValue(op.getResult()));
    printLoadOp(op);
    os << ".clone());\n";
  }

  // If any uses are outside the levelled op, also assign it it's SSA value.
  if (usedByNonLevelledOp(op) || !useLevels_) {
    emitAssignPrefix(op.getResult());
    bool isRef =
        isa<tfhe_rust::TfheRustDialect>(op.getResult().getType().getDialect());
    bool storeUse = llvm::all_of(op.getResult().getUsers(), [](Operation *op) {
      return isa<memref::StoreOp>(*op);
    });
    os << ((isRef && !storeUse) ? "&" : "");
    printLoadOp(op);
    os << ";\n";
  }

  return success();
}

FailureOr<std::string> TfheRustEmitter::convertType(Type type) {
  // Note: these are probably not the right type names to use exactly, and
  // they will need to chance to the right values once we try to compile it
  // against a specific API version.
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
        return (type.isUnsigned() ? std::string("u") : "") + "i" +
               std::to_string(width.value());
      })
      // TODO(#474): Generalize to any encrypted uint.
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
      // TODO(#474): Generalize to any encrypted uint.
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
                                 SelectVariableNames *variableNames,
                                 bool useLevels)
    : useLevels_(useLevels), os(os), variableNames(variableNames) {}
}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir
