#include "lib/Transforms/YosysOptimizer/RTLILImporter.h"

#include <cassert>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "lib/Dialect/Comb/IR/CombOps.h"
#include "llvm/include/llvm/ADT/MapVector.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/FoldUtils.h"      // from @llvm-project

// Block clang-format from reordering
// clang-format off
#include "kernel/yosys.h" // from @at_clifford_yosys
// clang-format on

#define DEBUG_TYPE "rtlil-importer"

namespace mlir {
namespace heir {

using ::Yosys::RTLIL::Module;
using ::Yosys::RTLIL::SigSpec;
using ::Yosys::RTLIL::Wire;

namespace {

// getTypeForWire gets the MLIR type corresponding to the RTLIL wire. If the
// wire is an integer with multiple bits, then the MLIR type is a memref of
// bits.
Type getTypeForWire(OpBuilder &b, Wire *wire) {
  auto intTy = b.getI1Type();
  if (wire->width == 1) {
    return intTy;
  }
  return MemRefType::get({wire->width}, intTy);
}

}  // namespace

llvm::SmallVector<std::string, 10> getTopologicalOrder(
    std::stringstream &torderOutput) {
  llvm::SmallVector<std::string, 10> cells;
  std::string line;
  while (std::getline(torderOutput, line)) {
    auto lineCell = line.find("cell ");
    if (lineCell != std::string::npos) {
      cells.push_back(Yosys::RTLIL::escape_id(
          line.substr(lineCell + 5, std::string::npos)));
    }
  }
  return cells;
}

void RTLILImporter::addWireValue(Wire *wire, Value value) {
  wireNameToValue.insert(std::make_pair(wire->name.str(), value));
}

Value RTLILImporter::getWireValue(Wire *wire) {
  auto wireName = wire->name.str();
  assert(wireNameToValue.contains(wireName));
  return wireNameToValue.at(wireName);
}

Value RTLILImporter::getBit(
    const SigSpec &conn, ImplicitLocOpBuilder &b,
    llvm::MapVector<Wire *, SmallVector<Value>> &retBitValues) {
  // Because the cells are in topological order, and Yosys should have
  // removed redundant wire-wire mappings, the cell's inputs must be a bit
  // of an input wire, in the map of already defined wires (which are
  // bits), or a constant bit.
  assert(conn.is_wire() || conn.is_fully_const() || conn.is_bit());
  if (conn.is_wire()) {
    auto name = conn.as_wire()->name.str();
    assert(wireNameToValue.contains(name));
    return wireNameToValue[name];
  }
  if (conn.is_fully_const()) {
    auto bit = conn.as_const();
    auto constantOp = b.createOrFold<arith::ConstantOp>(
        b.getIntegerAttr(b.getIntegerType(1), bit.as_int()));
    return constantOp;
  }
  // Extract the bit of the multi-bit input or output wire.
  assert(conn.as_bit().is_wire());
  auto bit = conn.as_bit();
  if (retBitValues.contains(bit.wire)) {
    return retBitValues[bit.wire][bit.offset];
  }
  auto argA = getWireValue(bit.wire);
  auto extractOp = b.create<memref::LoadOp>(
      argA, b.create<arith::ConstantIndexOp>(bit.offset).getResult());
  return extractOp;
}

void RTLILImporter::addResultBit(
    const SigSpec &conn, Value result,
    llvm::MapVector<Wire *, SmallVector<Value>> &retBitValues) {
  if (!conn.is_wire() && !conn.is_bit()) {
    LLVM_DEBUG(llvm::errs()
               << "expected output connection to be an output wire or bit,"
               << " but got: chunk: " << conn.is_chunk());
    llvm_unreachable("expected output connection to be an output wire or bit,");
  }
  if (conn.is_wire()) {
    addWireValue(conn.as_wire(), result);
    return;
  }
  // This must be a bit of the multi-bit output wire.
  auto bit = conn.as_bit();
  assert(bit.is_wire() && retBitValues.contains(bit.wire));
  retBitValues[bit.wire][bit.offset] = result;
}

func::FuncOp RTLILImporter::importModule(
    Module *module, const SmallVector<std::string, 10> &cellOrdering) {
  // Gather input and output wires of the module to match up with the block
  // arguments.
  std::map<int, Wire *> wireArgs;
  std::map<int, Wire *> wireRet;

  OpBuilder builder(context);
  // Maintain a map from RTLIL output wires to the Values that comprise it
  // in order to reconstruct the multi-bit output.
  llvm::MapVector<Wire *, SmallVector<Value>> retBitValues;
  for (auto *wire : module->wires()) {
    // The RTLIL module may also have intermediate wires that are neither inputs
    // nor outputs.
    if (wire->port_input) {
      wireArgs[wire->port_id - 1] = wire;
    } else if (wire->port_output) {
      // These are indexed after the input wires.
      wireRet[wire->port_id - 1] = wire;
      retBitValues[wire].resize(wire->width);
    }
  }

  // Get ordered argument and return type lists.
  int numInputs = wireArgs.size();
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> retTypes;
  for (auto &[_, wireArg] : wireArgs) {
    argTypes.push_back(getTypeForWire(builder, wireArg));
  }
  for (auto &[_, wireReg] : wireRet) {
    retTypes.push_back(getTypeForWire(builder, wireReg));
  }

  // Build function.
  FunctionType funcType = builder.getFunctionType(argTypes, retTypes);
  auto function = func::FuncOp::create(
      builder.getUnknownLoc(), module->name.str().replace(0, 1, ""), funcType);
  function.setPrivate();

  auto *block = function.addEntryBlock();
  auto b = ImplicitLocOpBuilder::atBlockBegin(function.getLoc(), block);

  // Set the return bits to default 0
  auto constantOp = b.createOrFold<arith::ConstantOp>(
      b.getIntegerAttr(b.getIntegerType(1), 0));
  for (auto &[wire, values] : retBitValues) {
    values.assign(wire->width, constantOp);
  }

  // Map the RTLIL wires to the block arguments' Values.
  for (unsigned i = 0; i < wireArgs.size(); i++) {
    addWireValue(wireArgs[i], block->getArgument(i));
  }

  // Convert cells to Operations according to topological order.
  for (const auto &cellName : cellOrdering) {
    assert(module->cells_.count(cellName) != 0 &&
           "expected cell in RTLIL design");
    auto *cell = module->cells_[cellName];

    SmallVector<Value> inputValues;
    for (const auto &conn : getInputs(cell)) {
      inputValues.push_back(getBit(conn, b, retBitValues));
    }
    auto *op = createOp(cell, inputValues, b);
    auto resultConn = getOutput(cell);
    addResultBit(resultConn, op->getResult(0), retBitValues);
  }

  // Wire up remaining connections.
  for (const auto &conn : module->connections()) {
    auto output = conn.first;
    // These must be output wire connections (either an output bit or a bit of a
    // multi-bit output wire).
    assert((output.is_wire() ||
            (output.is_chunk() && output.as_chunk().is_wire()) ||
            (output.is_bit() && output.as_bit().is_wire()) ||
            llvm::all_of(output.chunks(),
                         [](const auto &chunk) { return chunk.is_wire(); })) &&
           "expected output to be a wire, chunk, or bit of a wire");
    if ((output.is_chunk() && !output.is_wire()) ||
        ((conn.second.is_chunk() && !conn.second.is_wire()) ||
         conn.second.chunks().size() > 1)) {
      // If one of the RHS or LHS is a chunk of a wire (and not a whole wire) OR
      // contains multiple chunks, then iterate bit by bit to assign the result
      // bits.
      for (auto i = 0; i < output.size(); i++) {
        Value connValue = getBit(conn.second.bits().at(i), b, retBitValues);
        addResultBit(output.bits().at(i), connValue, retBitValues);
      }
    } else {
      // This may be a single bit, a chunk of a wire, or a whole wire.
      Value connValue = getBit(conn.second, b, retBitValues);
      addResultBit(output, connValue, retBitValues);
    }
  }

  // Concatenate result bits if needed, and return result.
  SmallVector<Value, 4> returnValues;
  for (unsigned i = 0; i < retTypes.size(); i++) {
    auto *resultWire = wireRet[numInputs + i];  // Indexed after input ports
    auto retBits = retBitValues[resultWire];
    // If we are returning a whole wire as is (e.g. the input wire) or a single
    // bit, we do not need to concat any return bits.
    if (wireNameToValue.contains(resultWire->name.str())) {
      returnValues.push_back(getWireValue(resultWire));
    } else {
      // We are in a multi-bit scenario.
      assert(retBits.size() > 1);
      auto allocOp = b.create<memref::AllocOp>(
          cast<MemRefType>(getTypeForWire(b, resultWire)));
      for (unsigned j = 0; j < retBits.size(); j++) {
        b.create<memref::StoreOp>(
            retBits[j], allocOp.getResult(),
            ValueRange{b.create<arith::ConstantIndexOp>(j)});
      }
      returnValues.push_back(allocOp.getResult());
    }
  }
  b.create<func::ReturnOp>(returnValues);

  return function;
}

}  // namespace heir
}  // namespace mlir
