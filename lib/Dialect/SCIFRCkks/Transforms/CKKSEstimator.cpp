#include "lib/Dialect/SCIFRCkks/Transforms/CKKSEstimator.h"

#include <iomanip>
#include <string>

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/SCIFRBool/Transforms/PerfCounter.h"
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
#include "lib/Dialect/SCIFRCkks/Transforms/Passes.h.inc"
// clang-format on

#define DEBUG_ON 0

namespace mlir {
namespace heir {

void CKKSLongestChain::reset() {
  node_indexes.clear();
  node_names.clear();
  opcodes.clear();
}

void CKKSLongestChain::print() const {
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
    llvm::outs() << " " << CKKSOpsData::opcode_name(opcode);
  }

  llvm::outs() << "\n";
}

void CKKSOpsData::reset() {
  core_counts.clear();
  core_counts.resize(CKKSOpcode::UNKNOWN_CKKS, 0);
  ops_count.clear();
  ops_count.resize(CKKSOpcode::UNKNOWN_CKKS, 0);

  n_accumulated_throughput = 0;
  nodes.clear();
  adjacencies.clear();
  longest_chain.reset();
};

std::string CKKSOpsData::opcode_name(CKKSOpcode opcode) {
  switch (opcode) {
    case ADD:
      return "ADD";
    case ADDPLAIN:
      return "ADDPLAIN";
    case SUB:
      return "SUB";
    case SUBPLAIN:
      return "SUBPLAIN";
    case MUL:
      return "MUL";
    case MULPLAIN:
      return "MULPLAIN";
    case ROTATE:
      return "ROTATE";
    case EXTRACT:
      return "EXTRACT";
    case NEGATE:
      return "NEGATE";
    case RELINEARIZE:
      return "RELINEARIZE";
    case RESCALE:
      return "RESCALE";
    case UNKNOWN_CKKS:
      return "UNKNOWN";
  };
  return "UNKNOWN";
}

std::vector<CKKSOpcode> CKKSOpsData::opcodes() {
  return {
      CKKSOpcode::ADD,         CKKSOpcode::ADDPLAIN, CKKSOpcode::SUB,
      CKKSOpcode::SUBPLAIN,    CKKSOpcode::MUL,      CKKSOpcode::MULPLAIN,
      CKKSOpcode::ROTATE,      CKKSOpcode::EXTRACT,  CKKSOpcode::NEGATE,
      CKKSOpcode::RELINEARIZE, CKKSOpcode::RESCALE,  CKKSOpcode::UNKNOWN_CKKS,
  };
}

int CKKSOpsData::get_core_counts(CKKSOpcode opcode) {
  const auto& count_and_th = get_core_count_and_throughput(opcode);
  return count_and_th[0];
}

int CKKSOpsData::get_throughput(CKKSOpcode opcode) {
  const auto& count_and_th = get_core_count_and_throughput(opcode);
  return count_and_th[1];
}

int CKKSOpsData::get_latency(CKKSOpcode opcode) {
  const auto& count_and_th = get_core_count_and_throughput(opcode);
  return BATCH_SIZE * count_and_th[1];
}

std::vector<int> CKKSOpsData::get_core_count_and_throughput(CKKSOpcode opcode) {
  switch (opcode) {
    case ADD:
    case ADDPLAIN:
      return {2, 4};
    case SUB:
    case SUBPLAIN:
      return {2, 4};
    case MUL:
    case MULPLAIN:
      return {53000, 4};
    case NEGATE:
      return {1, 1};
    case ROTATE:
      return {300, 5};
    case RELINEARIZE:
      return {600, 5};
    case RESCALE:
      return {4, 4};
    case EXTRACT:
      return {4, 4};
    case UNKNOWN_CKKS:
      return {0, 0};
      // "MUL_RAW": return {53000, 4};
      // "META_SHAPE": return {0, 0};
      // "PMUL": return {53000, 2};
      // "READ": return {2, 4};
      // "LABEL": return {0, 0};
      // "BOOTSTRAP": return {1200, 4};
      // "SQ": return {2, 2};
      // "OUTPUT": return {0, 0};
      // "PSUB": return {0, 0};
      // "PARAM": return {0, 0};
      // "BOOTSTRAPREAL": return {0, 0};
      // "SET": return {0, 0};
      // "SET_CI": return {0, 0};
      // "SELECTOR": return {0, 0};
      // "ADD_SCALAR_INT": return {0, 0};
      // "ADD_SCALAR_DOUBLE": return {2, 2};
      // "MUL_SCALAR_DOUBLE": return {2, 2};
      // "OUTPUT": return {0, 0};
      // "None": return {0, 0};
  }

  return {0, 0};
}

void CKKSOpsData::display() const { display(opcodes()); }

void CKKSOpsData::display(const std::vector<CKKSOpcode>& opcodes) const {
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
    if (opcode == CKKSOpcode::UNKNOWN_CKKS) continue;
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

cornami::DialectResourceMap CKKSOpsData::get_resource_map() const {
  cornami::DialectResourceMap ruse;  // resource use
  ruse["AND"] = 0;

  return ruse;
}
}  // namespace heir

namespace cornami {
#define GEN_PASS_DEF_CKKSESTIMATOR
#include "lib/Dialect/SCIFRCkks/Transforms/Passes.h.inc"

using namespace mlir::heir;

struct CKKSEstimator : impl::CKKSEstimatorBase<CKKSEstimator> {
  using CKKSEstimatorBase::CKKSEstimatorBase;
  PerfCounter m_data;
  CKKSOpsData m_ckksops;

  std::string getOperationAsString(mlir::Operation* op) {
    std::string opString;
    llvm::raw_string_ostream ss(opString);
    op->print(ss);
    return ss.str();
  }

  CKKSOpcode getOpcode(Operation* op) {
    CKKSOpcode opcode = CKKSOpcode::UNKNOWN_CKKS;
    if (llvm::isa<ckks::AddOp>(op)) {
      opcode = CKKSOpcode::ADD;
    } else if (llvm::isa<ckks::AddPlainOp>(op)) {
      opcode = CKKSOpcode::ADDPLAIN;
    } else if (llvm::isa<ckks::SubOp>(op)) {
      opcode = CKKSOpcode::SUB;
    } else if (llvm::isa<ckks::SubPlainOp>(op)) {
      opcode = CKKSOpcode::SUBPLAIN;
    } else if (llvm::isa<ckks::MulOp>(op)) {
      opcode = CKKSOpcode::MUL;
    } else if (llvm::isa<ckks::MulPlainOp>(op)) {
      opcode = CKKSOpcode::MULPLAIN;
    } else if (llvm::isa<ckks::RotateOp>(op)) {
      opcode = CKKSOpcode::ROTATE;
      //} else if  ( llvm::isa<ckks::ExtractOp>(op) ) {
      // opcode = CKKSOpcode::EXTRACT;
    } else if (llvm::isa<ckks::NegateOp>(op)) {
      opcode = CKKSOpcode::NEGATE;
    } else if (llvm::isa<ckks::RelinearizeOp>(op)) {
      opcode = CKKSOpcode::RELINEARIZE;
    } else if (llvm::isa<ckks::RescaleOp>(op)) {
      // m_ckksops.n_rescaleop++;
      opcode = CKKSOpcode::RESCALE;
    }
    return opcode;
  }

  void runOnOperation() override {
    // MLIRContext &context = getContext();
    graph::Graph<Operation*> graph;
    m_ckksops.reset();

    getOperation()->walk<WalkOrder::PostOrder>([&](Operation* op) {
      CKKSOpcode opcode = getOpcode(op);
      if (opcode == CKKSOpcode::UNKNOWN_CKKS) {
        op->emitWarning() << "Unknown operator not counted for Estimate\n";
      } else {
        m_ckksops.ops_count[opcode]++;
        m_ckksops.core_counts[opcode] += CKKSOpsData::get_core_counts(opcode);
        m_ckksops.n_accumulated_throughput +=
            CKKSOpsData::get_throughput(opcode);
      }

      // op should be CKKS operator only
      graph.addVertex(op);
      if (op->getNumOperands() > 0) {
        llvm::outs() << "Operator : "
                     << op->getName().getIdentifier().getValue().str()
                     << " has " << op->getNumOperands() << " operands.\n";
        for (auto inoperand : op->getOperands()) {
          if (!inoperand.getDefiningOp()) continue;
          graph.addVertex(inoperand.getDefiningOp());
          graph.addEdge(inoperand.getDefiningOp(), op,
                        CKKSOpsData::get_latency(opcode));
        }
      }

      llvm::outs() << "operation = " << "\n";
      op->dump();
      llvm::outs() << "\n";
    });

    m_ckksops.display();

    dump(graph);
    auto cp_ = graph.findApproximateCriticalPath();
    if (!succeeded(cp_) || cp_.value().empty()) {
      llvm::outs() << "======= Cannot find useful CP =========\n";
    } else {
      llvm::outs() << "======= STATS ALONG CP =========\n";

      m_ckksops.reset();
      for (const auto& op : cp_.value()) {
        CKKSOpcode opcode = getOpcode(op);
        if (opcode == CKKSOpcode::UNKNOWN_CKKS) {
          op->emitWarning() << "Unknown operator not counted for Estimate\n";
        } else {
          m_ckksops.ops_count[opcode]++;
          m_ckksops.core_counts[opcode] += CKKSOpsData::get_core_counts(opcode);
          m_ckksops.n_accumulated_throughput +=
              CKKSOpsData::get_throughput(opcode);
        }
      }

      if (DEBUG_ON) {
        for (const auto& op : cp_.value()) {
          CKKSOpcode opcode = getOpcode(op);
          llvm::outs() << CKKSOpsData::opcode_name(opcode) << "\n";
        }
      }

      llvm::outs() << "\n";

      m_ckksops.display();
    }
    llvm::outs() << "\n";
  }
};  // struct CKKSEstimator

}  // namespace cornami
}  // namespace mlir
