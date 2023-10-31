#include "lib/Transforms/YosysOptimizer/RTLILImporter.h"

#include <cassert>
#include <sstream>
#include <string>
#include <utility>

#include "include/Dialect/Comb/IR/CombOps.h"
#include "kernel/rtlil.h"                     // from @at_clifford_yosys
#include "llvm/include/llvm/ADT/MapVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/FoldUtils.h"     // from @llvm-project

namespace mlir {
namespace heir {

using ::Yosys::RTLIL::Module;
using ::Yosys::RTLIL::SigSpec;
using ::Yosys::RTLIL::Wire;

llvm::SmallVector<std::string, 10> getTopologicalOrder(
    std::stringstream &torderOutput) {
  llvm::SmallVector<std::string, 10> cells;
  std::string line;
  while (std::getline(torderOutput, line)) {
    auto lineCell = line.find("cell ");
    if (lineCell != std::string::npos) {
      cells.push_back(line.substr(lineCell + 5, std::string::npos));
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
    auto offset = retBitValues[bit.wire].size() - bit.offset - 1;
    return retBitValues[bit.wire][offset];
  }
  auto argA = getWireValue(bit.wire);
  auto extractOp =
      b.createOrFold<comb::ExtractOp>(b.getI1Type(), argA, bit.offset);
  return extractOp;
}

void RTLILImporter::addResultBit(
    const SigSpec &conn, Value result,
    llvm::MapVector<Wire *, SmallVector<Value>> &retBitValues) {
  assert(conn.is_wire() || conn.is_bit());
  if (conn.is_wire()) {
    addWireValue(conn.as_wire(), result);
    return;
  }
  // This must be a bit of the multi-bit output wire.
  auto bit = conn.as_bit();
  assert(bit.is_wire() && retBitValues.contains(bit.wire));
  auto offset = retBitValues[bit.wire].size() - bit.offset - 1;
  retBitValues[bit.wire][offset] = result;
}

func::FuncOp RTLILImporter::importModule(
    Module *module, const SmallVector<std::string, 10> &cellOrdering) {
  // Gather input and output wires of the module to match up with the block
  // arguments.
  SmallVector<Type, 4> argTypes;
  SmallVector<Wire *, 4> wireArgs;
  SmallVector<Type, 4> retTypes;
  SmallVector<Wire *, 4> wireRet;

  OpBuilder builder(context);
  // Maintain a map from RTLIL output wires to the Values that comprise it
  // in order to reconstruct the multi-bit output.
  llvm::MapVector<Wire *, SmallVector<Value>> retBitValues;
  for (auto *wire : module->wires()) {
    // The RTLIL module may also have intermediate wires that are neither inputs
    // nor outputs.
    if (wire->port_input) {
      argTypes.push_back(builder.getIntegerType(wire->width));
      wireArgs.push_back(wire);
    } else if (wire->port_output) {
      retTypes.push_back(builder.getIntegerType(wire->width));
      wireRet.push_back(wire);
      retBitValues[wire].resize(wire->width);
    }
  }

  // Build function.
  // TODO(https://github.com/google/heir/issues/111): Pass in data to fix
  // function location.
  FunctionType funcType = builder.getFunctionType(argTypes, retTypes);
  auto function = func::FuncOp::create(
      builder.getUnknownLoc(), module->name.str().replace(0, 1, ""), funcType);
  function.setPrivate();

  auto *block = function.addEntryBlock();
  auto b = ImplicitLocOpBuilder::atBlockBegin(function.getLoc(), block);

  // Map the RTLIL wires to the block arguments' Values.
  for (auto i = 0; i < wireArgs.size(); i++) {
    addWireValue(wireArgs[i], block->getArgument(i));
  }

  // Convert cells to Operations according to topological order.
  for (const auto &cellName : cellOrdering) {
    assert(module->cells_.count("\\" + cellName) != 0 &&
           "expected cell in RTLIL design");
    auto *cell = module->cells_["\\" + cellName];

    SmallVector<Value, 4> inputValues;
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
    assert(output.is_wire() || output.as_bit().is_wire());
    assert(retBitValues.contains(output.as_wire()) ||
           retBitValues.contains(output.as_bit().wire));
    if (conn.second.chunks().size() == 1) {
      // This may be a single bit, or it may be a whole wire.
      Value connValue = getBit(conn.second, b, retBitValues);
      addResultBit(output, connValue, retBitValues);
    } else {
      auto chunks = conn.second.chunks();
      for (auto i = 0; i < output.size(); i++) {
        Value connValue = getBit(conn.second.bits().at(i), b, retBitValues);
        addResultBit(output.bits().at(i), connValue, retBitValues);
      }
    }
  }

  // Concatenate result bits if needed, and return result.
  SmallVector<Value, 4> returnValues;
  for (const auto &[resultWire, retBits] : retBitValues) {
    // If we are returning a whole wire as is (e.g. the input wire) or a single
    // bit, we do not need to concat any return bits.
    if (wireNameToValue.contains(resultWire->name.str())) {
      returnValues.push_back(getWireValue(resultWire));
    } else {
      // We are in a multi-bit scenario.
      assert(retBits.size() > 1);
      auto concatOp = b.create<comb::ConcatOp>(retBits);
      returnValues.push_back(concatOp.getResult());
    }
  }
  b.create<func::ReturnOp>(returnValues);

  return function;
}

}  // namespace heir
}  // namespace mlir
