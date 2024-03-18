#include "lib/Transforms/YosysOptimizer/BooleanGateImporter.h"

#include <cassert>

#include "include/Dialect/Comb/IR/CombOps.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"    // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

// Block clang-format from reordering
// clang-format off
#include "kernel/rtlil.h" // from @at_clifford_yosys
// clang-format on

namespace mlir {
namespace heir {

mlir::Operation *BooleanGateImporter::createOp(Yosys::RTLIL::Cell *cell,
                                               SmallVector<Value> &inputs,
                                               ImplicitLocOpBuilder &b) const {
  auto op = llvm::StringSwitch<mlir::Operation *>(cell->type.substr(1))
                .Case("inv", b.create<comb::InvOp>(inputs[0], false))
                .Case("xnor2", b.create<comb::XNorOp>(inputs, false))
                .Case("and2", b.create<comb::AndOp>(inputs, false))
                .Case("xor2", b.create<comb::XorOp>(inputs, false))
                .Case("nand2", b.create<comb::NandOp>(inputs, false))
                .Case("nor2", b.create<comb::NorOp>(inputs, false))
                .Case("or2", b.create<comb::OrOp>(inputs, false))
                .Default(nullptr);
  if (op == nullptr) {
    llvm_unreachable("unexpected cell type");
  }
  return op;
}

SmallVector<Yosys::RTLIL::SigSpec> BooleanGateImporter::getInputs(
    Yosys::RTLIL::Cell *cell) const {
  // Return all non-Y named attributes.
  SmallVector<Yosys::RTLIL::SigSpec, 4> inputs;
  for (auto &conn : cell->connections()) {
    if (conn.first.contains("Y")) {
      continue;
    }
    inputs.push_back(conn.second);
  }

  return inputs;
}

Yosys::RTLIL::SigSpec BooleanGateImporter::getOutput(
    Yosys::RTLIL::Cell *cell) const {
  return cell->getPort(Yosys::RTLIL::IdString("\\Y"));
}

}  // namespace heir
}  // namespace mlir
