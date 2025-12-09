#include "lib/Transforms/YosysOptimizer/BooleanGateImporter.h"

#include <cassert>

#include "lib/Dialect/Comb/IR/CombOps.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"    // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

// Block clang-format from reordering
// clang-format off
#include "kernel/rtlil.h"  // from @at_clifford_yosys
// clang-format on

namespace mlir {
namespace heir {

mlir::Operation* BooleanGateImporter::createOp(Yosys::RTLIL::Cell* cell,
                                               SmallVector<Value>& inputs,
                                               ImplicitLocOpBuilder& b) const {
  // standard cell names look like $_CELL_
  auto* op = llvm::StringSwitch<mlir::Operation*>(
                 cell->type.substr(2, cell->type.size() - 3))
                 .Case("NOT", comb::InvOp::create(b, inputs[0], false))
                 .Case("XNOR", comb::XNorOp::create(b, inputs, false))
                 .Case("AND", comb::AndOp::create(b, inputs, false))
                 .Case("XOR", comb::XorOp::create(b, inputs, false))
                 .Case("NAND", comb::NandOp::create(b, inputs, false))
                 .Case("NOR", comb::NorOp::create(b, inputs, false))
                 .Case("OR", comb::OrOp::create(b, inputs, false))
                 .Default(nullptr);
  if (op == nullptr) {
    llvm_unreachable("unexpected cell type");
  }
  return op;
}

SmallVector<Yosys::RTLIL::SigSpec> BooleanGateImporter::getInputs(
    Yosys::RTLIL::Cell* cell) const {
  // Return all non-Y named attributes.
  SmallVector<Yosys::RTLIL::SigSpec> inputs;
  for (auto& conn : cell->connections()) {
    if (conn.first.contains("Y")) {
      continue;
    }
    inputs.push_back(conn.second);
  }

  return inputs;
}

Yosys::RTLIL::SigSpec BooleanGateImporter::getOutput(
    Yosys::RTLIL::Cell* cell) const {
  return cell->getPort(Yosys::RTLIL::IdString("\\Y"));
}

}  // namespace heir
}  // namespace mlir
