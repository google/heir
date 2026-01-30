#include "lib/Backend/cornami/Dialect/SCIFRCkks/Analysis/OpenfheEstimator.h"

#include <iomanip>
#include <string>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/TopologicalSortUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

// clang-format off
#include "lib/Backend/cornami/Dialect/SCIFRCkks/Analysis/Passes.h.inc"
// clang-format on

#define DEBUG_ON 0

namespace mlir {
namespace heir {

void OpenfheLongestChain::reset() {
  node_indexes.clear();
  node_names.clear();
  opcodes.clear();
}

void OpenfheLongestChain::print() const {
  if (DEBUG_ON) {
    llvm::outs() << "longest chain \n";
    llvm::outs() << "longest node index seq \n";

    for (const auto& index : node_indexes) {
      llvm::outs() << " " << index;
    }
    llvm::outs() << "\n";

    llvm::outs() << "longest node name seq \n";

    for (const auto& name : node_names) {
      llvm::outs() << "\n\n" << name;
    }
  }

  llvm::outs() << "\n\nlongest opcode seq \n";

  for (const auto& opcode : opcodes) {
    llvm::outs() << " " << OpenfheOpsData::opcode_name(opcode);
  }

  llvm::outs() << "\n";
}

void OpenfheOpsData::reset() {
  core_counts.clear();
  core_counts.resize(OpenfheOpcode::UNKNOWN_OPENFHE, 0);
  ops_count.clear();
  ops_count.resize(OpenfheOpcode::UNKNOWN_OPENFHE, 0);
  n_accumulated_throughput = 0;
  nodes.clear();
  adjacencies.clear();
  longest_chain.reset();
};

std::string OpenfheOpsData::opcode_name(OpenfheOpcode opcode) {
  switch (opcode) {
    case AddOp:
      return "AddOp";
    case AutomorphOp:
      return "AutomorphOp";
    case DecryptOp:
      return "DecryptOp";
    case EncryptOp:
      return "EncryptOp";
    case GenContextOp:
      return "GenContextOp";
    case GenMulKeyOp:
      return "GenMulKeyOp";
    case GenParamsOp:
      return "GenParamsOp";
    case GenRotKeyOp:
      return "GenRotKeyOp";
    case KeySwitchOp:
      return "KeySwitchOp";
    case LevelReduceOp:
      return "LevelReduceOp";
    case ModReduceOp:
      return "ModReduceOp";
    case MulConstOp:
      return "MulConstOp";
    case MulNoRelinOp:
      return "MulNoRelinOp";
    case MulOp:
      return "MulOp";
    case MulPlainOp:
      return "MulPlainOp";
    case NegateOp:
      return "NegateOp";
    case RelinOp:
      return "RelinOp";
    case RotOp:
      return "RotOp";
    case SquareOp:
      return "SquareOp";
    case SubOp:
      return "SubOp";
    case UNKNOWN_OPENFHE:
      return "UNKNOWN_OPENFHE";
  };
  return "UNKNOWN_OPENFHE";
}

std::vector<OpenfheOpcode> OpenfheOpsData::opcodes() {
  return {OpenfheOpcode::AddOp,          OpenfheOpcode::AutomorphOp,
          OpenfheOpcode::DecryptOp,      OpenfheOpcode::EncryptOp,
          OpenfheOpcode::GenContextOp,   OpenfheOpcode::GenMulKeyOp,
          OpenfheOpcode::GenParamsOp,    OpenfheOpcode::GenRotKeyOp,
          OpenfheOpcode::KeySwitchOp,    OpenfheOpcode::LevelReduceOp,
          OpenfheOpcode::ModReduceOp,    OpenfheOpcode::MulConstOp,
          OpenfheOpcode::MulNoRelinOp,   OpenfheOpcode::MulOp,
          OpenfheOpcode::MulPlainOp,     OpenfheOpcode::NegateOp,
          OpenfheOpcode::RelinOp,        OpenfheOpcode::RotOp,
          OpenfheOpcode::SquareOp,       OpenfheOpcode::SubOp,
          OpenfheOpcode::UNKNOWN_OPENFHE};
}

int OpenfheOpsData::get_core_counts(OpenfheOpcode opcode) {
  const auto& count_and_th = get_core_count_and_throughput(opcode);
  return count_and_th[0];
}

int OpenfheOpsData::get_throughput(OpenfheOpcode opcode) {
  const auto& count_and_th = get_core_count_and_throughput(opcode);
  return count_and_th[1];
}

int OpenfheOpsData::get_latency(OpenfheOpcode opcode) {
  const auto& count_and_th = get_core_count_and_throughput(opcode);
  return BATCH_SIZE * count_and_th[1];
}

std::vector<int> OpenfheOpsData::get_core_count_and_throughput(
    OpenfheOpcode opcode) {
  switch (opcode) {
    case AddOp:
      return {2, 4};
    case AutomorphOp:
      return {2, 4};
    case DecryptOp:
      return {2, 4};
    case EncryptOp:
      return {2, 4};
    case GenContextOp:
      return {2, 4};
    case GenMulKeyOp:
      return {2, 4};
    case GenParamsOp:
      return {2, 4};
    case GenRotKeyOp:
      return {2, 4};
    case KeySwitchOp:
      return {2, 4};
    case LevelReduceOp:
      return {2, 4};
    case ModReduceOp:
      return {2, 4};
    case MulConstOp:
      return {2, 4};
    case MulNoRelinOp:
      return {2, 4};
    case MulOp:
      return {2, 4};
    case MulPlainOp:
      return {2, 4};
    case NegateOp:
      return {2, 4};
    case RelinOp:
      return {2, 4};
    case RotOp:
      return {2, 4};
    case SquareOp:
      return {2, 4};
    case SubOp:
      return {2, 4};
    case UNKNOWN_OPENFHE:
      return {0, 0};
  }

  return {0, 0};
}

void OpenfheOpsData::display() const { display(opcodes()); }

void OpenfheOpsData::display(const std::vector<OpenfheOpcode>& opcodes) const {
  auto set_w_str = [](int width, const std::string& s) -> std::string {
    std::stringstream ss;
    ss.width(width);
    ss << s;
    return ss.str();
  };

  auto set_w_int = [](int width, int value) -> std::string {
    std::stringstream ss;
    ss.width(width);
    ss << value;
    return ss.str();
  };

  auto set_w_double = [](int width, double value) -> std::string {
    std::stringstream ss;
    ss.width(width);
    ss.precision(2);
    ss << value;
    return ss.str();
  };

  int cumulative_fcs = 0;
  for (const auto& opcode : opcodes) {
    if (opcode == OpenfheOpcode::UNKNOWN_OPENFHE) continue;
    cumulative_fcs += core_counts[opcode];
    llvm::outs() << set_w_str(15, opcode_name(opcode)) << " : "
                 << set_w_int(6, ops_count[opcode])
                 << " | FC = " << set_w_int(6, core_counts[opcode])
                 << " LATENCY = "
                 << set_w_double(6, BATCH_SIZE * get_throughput(opcode))
                 << "\n";
  }

  double cumulative_latency = (BATCH_SIZE * n_accumulated_throughput);
  const uint64_t fcs_per_chip = 2048;
  uint64_t chip_count = (cumulative_fcs + fcs_per_chip - 1) / fcs_per_chip;

  llvm::outs() << "---------------------------------------------\n";
  llvm::outs() << "Total FC = " << cumulative_fcs
               << " | LATENCY = " << set_w_double(4, cumulative_latency)
               << " | CHIPS = " << chip_count << "\n";
}
}  // namespace heir
namespace cornami {
#define GEN_PASS_DEF_OPENFHEESTIMATOR
#include "lib/Backend/cornami/Dialect/SCIFRCkks/Analysis/Passes.h.inc"

using namespace mlir::heir;

struct OpenfheEstimator : impl::OpenfheEstimatorBase<OpenfheEstimator> {
  using OpenfheEstimatorBase::OpenfheEstimatorBase;

  OpenfheOpsData m_openfheops;

  std::string getOperationAsString(mlir::Operation* op) {
    std::string opString;
    llvm::raw_string_ostream ss(opString);
    op->print(ss);
    return ss.str();
  }

  OpenfheOpcode getOpcode(Operation* op) {
    OpenfheOpcode opcode = OpenfheOpcode::UNKNOWN_OPENFHE;
    if (llvm::isa<openfhe::AddOp>(op)) return OpenfheOpcode::AddOp;
    if (llvm::isa<openfhe::AutomorphOp>(op)) return OpenfheOpcode::AutomorphOp;
    if (llvm::isa<openfhe::DecryptOp>(op)) return OpenfheOpcode::DecryptOp;
    if (llvm::isa<openfhe::EncryptOp>(op)) return OpenfheOpcode::EncryptOp;
    if (llvm::isa<openfhe::GenContextOp>(op))
      return OpenfheOpcode::GenContextOp;
    if (llvm::isa<openfhe::GenMulKeyOp>(op)) return OpenfheOpcode::GenMulKeyOp;
    if (llvm::isa<openfhe::GenParamsOp>(op)) return OpenfheOpcode::GenParamsOp;
    if (llvm::isa<openfhe::GenRotKeyOp>(op)) return OpenfheOpcode::GenRotKeyOp;
    if (llvm::isa<openfhe::KeySwitchOp>(op)) return OpenfheOpcode::KeySwitchOp;
    if (llvm::isa<openfhe::LevelReduceOp>(op))
      return OpenfheOpcode::LevelReduceOp;
    if (llvm::isa<openfhe::ModReduceOp>(op)) return OpenfheOpcode::ModReduceOp;
    if (llvm::isa<openfhe::MulConstOp>(op)) return OpenfheOpcode::MulConstOp;
    if (llvm::isa<openfhe::MulNoRelinOp>(op))
      return OpenfheOpcode::MulNoRelinOp;
    if (llvm::isa<openfhe::MulOp>(op)) return OpenfheOpcode::MulOp;
    if (llvm::isa<openfhe::MulPlainOp>(op)) return OpenfheOpcode::MulPlainOp;
    if (llvm::isa<openfhe::NegateOp>(op)) return OpenfheOpcode::NegateOp;
    if (llvm::isa<openfhe::RelinOp>(op)) return OpenfheOpcode::RelinOp;
    if (llvm::isa<openfhe::RotOp>(op)) return OpenfheOpcode::RotOp;
    if (llvm::isa<openfhe::SquareOp>(op)) return OpenfheOpcode::SquareOp;
    if (llvm::isa<openfhe::SubOp>(op)) return OpenfheOpcode::SubOp;
    return opcode;
  }

  void runOnOperation() override {
    // MLIRContext &context = getContext();
    graph::Graph<Operation*> graph;
    m_openfheops.reset();

    getOperation()->walk<WalkOrder::PostOrder>([&](Operation* op) {
      OpenfheOpcode opcode = getOpcode(op);
      if (opcode == OpenfheOpcode::UNKNOWN_OPENFHE) {
        op->emitWarning() << "Unknown operator not counted for Estimate\n";
      } else {
        m_openfheops.ops_count[opcode]++;
        m_openfheops.core_counts[opcode] +=
            OpenfheOpsData::get_core_counts(opcode);
        m_openfheops.n_accumulated_throughput +=
            OpenfheOpsData::get_throughput(opcode);
      }

      // op should be Openfhe operator only
      graph.addVertex(op);
      if (op->getNumOperands() > 0) {
        llvm::outs() << "Operator : "
                     << op->getName().getIdentifier().getValue().str()
                     << " has " << op->getNumOperands() << " operands.\n";
        for (auto inoperand : op->getOperands()) {
          if (!inoperand.getDefiningOp()) continue;
          graph.addVertex(inoperand.getDefiningOp());
          graph.addEdge(inoperand.getDefiningOp(), op,
                        OpenfheOpsData::get_latency(opcode));
        }
      }

      llvm::outs() << "operation = " << "\n";
      op->dump();
      llvm::outs() << "\n";
    });

    m_openfheops.display();

    dump(graph);
    auto cp_ = graph.findApproximateCriticalPath();
    if (!succeeded(cp_) || cp_.value().empty()) {
      llvm::outs() << "======= Cannot find useful CP =========\n";
    } else {
      llvm::outs() << "======= STATS ALONG CP =========\n";

      m_openfheops.reset();
      for (const auto& op : cp_.value()) {
        OpenfheOpcode opcode = getOpcode(op);
        if (opcode == OpenfheOpcode::UNKNOWN_OPENFHE) {
          op->emitWarning() << "Unknown operator not counted for Estimate\n";
        } else {
          m_openfheops.ops_count[opcode]++;
          m_openfheops.core_counts[opcode] +=
              OpenfheOpsData::get_core_counts(opcode);
          m_openfheops.n_accumulated_throughput +=
              OpenfheOpsData::get_throughput(opcode);
        }
      }

      if (DEBUG_ON) {
        for (const auto& op : cp_.value()) {
          OpenfheOpcode opcode = getOpcode(op);
          llvm::outs() << OpenfheOpsData::opcode_name(opcode) << "\n";
        }
      }

      llvm::outs() << "\n";

      m_openfheops.display();
    }
    llvm::outs() << "\n";
  }
};  // struct OpenfheEstimator

}  // namespace cornami
}  // namespace mlir
