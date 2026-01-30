#include "lib/Backend/cornami/Dialect/SCIFRBool/Analysis/CGGIEstimator.h"

#include <string>

#include "lib/Backend/cornami/Dialect/SCIFRBool/Analysis/PerfCounter.h"
#include "lib/Dialect/CGGI/IR/CGGIAttributes.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/TopologicalSortUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

// clang-format off
#include "lib/Backend/cornami/Dialect/SCIFRBool/Analysis/Passes.h.inc"
// clang-format on

namespace mlir {
namespace heir {
void CGGIOpsData::reset() {
  n_andop = {
      0,
  };  //::mlir::heir::cggi::AndOp;
  n_lut2op = {
      0,
  };  //::mlir::heir::cggi::Lut2Op;
  n_lut3op = {
      0,
  };  //::mlir::heir::cggi::Lut3Op,
  n_lutlincombop = {
      0,
  };  //::mlir::heir::cggi::LutLinCombOp,
  n_multilutlincombop = {
      0,
  };  //::mlir::heir::cggi::MultiLutLinCombOp,
  n_nandop = {
      0,
  };  //::mlir::heir::cggi::NandOp,
  n_norop = {
      0,
  };  //::mlir::heir::cggi::NorOp,
  n_notop = {
      0,
  };  //::mlir::heir::cggi::NotOp,
  n_orop = {
      0,
  };  //::mlir::heir::cggi::OrOp,
  n_packedop = {
      0,
  };  //::mlir::heir::cggi::PackedOp,
  n_xnorop = {
      0,
  };  //::mlir::heir::cggi::XNorOp,
  n_xorop = {
      0,
  };  //::mlir::heir::cggi::XorOp
};

void CGGIOpsData::display() const {
  uint64_t cumulative_fcs = 0, cumulative_latency = 0;

  for (auto& itr : get_resource_map()) {
    auto result = get_resource_use(to_enum(itr.first), itr.second);
    auto fcs = result.first;
    auto latency = result.second;
    llvm::outs() << itr.first << " : " << itr.second << " | FC = " << fcs
                 << " LATENCY = " << latency << "\n";
    cumulative_fcs += fcs;
    cumulative_latency += latency;
  }

  const uint64_t fcs_per_chip = 2048;
  uint64_t chip_count = (cumulative_fcs + fcs_per_chip - 1) / fcs_per_chip;
  llvm::outs() << "---------------------------------------------\n";
  llvm::outs() << "Total FC = " << cumulative_fcs
               << " | LATENCY = " << cumulative_latency
               << " | CHIPS = " << chip_count << "\n";
}

cornami::DialectResourceMap CGGIOpsData::get_resource_map() const {
  cornami::DialectResourceMap ruse;  // resource use
  ruse["AND"] = n_andop[0];
  ruse["LUT2"] = n_lut2op[0];
  ruse["LUT3"] = n_lut2op[0];
  ruse["LUTLINCOMB"] = n_lutlincombop[0];
  ruse["MULTILUTLINCOMB"] = n_multilutlincombop[0];
  ruse["NAND"] = n_nandop[0];
  ruse["NOR"] = n_norop[0];
  ruse["NOT"] = n_notop[0];
  ruse["OR"] = n_orop[0];
  ruse["PACKED"] = n_packedop[0];
  ruse["XNOR"] = n_xnorop[0];
  ruse["XOR"] = n_xorop[0];
  return ruse;
}

bool CGGIOpsData::analyze(mlir::Operation* op) {
  if (llvm::isa<cggi::AndOp>(op)) {
    this->n_andop[0]++;
  } else if (llvm::isa<cggi::Lut2Op>(op)) {
    this->n_lut2op[0]++;
  } else if (llvm::isa<cggi::Lut3Op>(op)) {
    this->n_lut3op[0]++;
  } else if (llvm::isa<cggi::LutLinCombOp>(op)) {
    this->n_lutlincombop[0]++;
  } else if (llvm::isa<cggi::MultiLutLinCombOp>(op)) {
    this->n_multilutlincombop[0]++;
  } else if (llvm::isa<cggi::NandOp>(op)) {
    this->n_nandop[0]++;
  } else if (llvm::isa<cggi::NotOp>(op)) {
    this->n_notop[0]++;
  } else if (llvm::isa<cggi::NorOp>(op)) {
    this->n_orop[0]++;
  } else if (llvm::isa<cggi::PackedOp>(op)) {
    this->n_packedop[0]++;
  } else if (llvm::isa<cggi::XNorOp>(op)) {
    this->n_xnorop[0]++;
  } else if (llvm::isa<cggi::XorOp>(op)) {
    this->n_xorop[0]++;
  } else {
    return false;
  }
  return true;
}
uint64_t CGGIOpsData::get_bootstrap_resource_fc() const { return 128; }

CGGIOpsData::CGGIOpsEnum CGGIOpsData::get_op_kind(mlir::Operation* op) {
  if (llvm::isa<cggi::AndOp>(op)) {
    return CGGIOpsEnum::AND;
  } else if (llvm::isa<cggi::Lut2Op>(op)) {
    return CGGIOpsEnum::LUT2;
  } else if (llvm::isa<cggi::Lut3Op>(op)) {
    return CGGIOpsEnum::LUT3;
  } else if (llvm::isa<cggi::LutLinCombOp>(op)) {
    return CGGIOpsEnum::LUTLINCOMB;
  } else if (llvm::isa<cggi::MultiLutLinCombOp>(op)) {
    return CGGIOpsEnum::MULTILUTLINCOMB;
  } else if (llvm::isa<cggi::NandOp>(op)) {
    return CGGIOpsEnum::NAND;
  } else if (llvm::isa<cggi::NotOp>(op)) {
    return CGGIOpsEnum::NOT;
  } else if (llvm::isa<cggi::NorOp>(op)) {
    return CGGIOpsEnum::NOR;
  } else if (llvm::isa<cggi::PackedOp>(op)) {
    return CGGIOpsEnum::PACKED;
  } else if (llvm::isa<cggi::XNorOp>(op)) {
    return CGGIOpsEnum::XNOR;
  } else if (llvm::isa<cggi::XorOp>(op)) {
    return CGGIOpsEnum::XOR;
  }
  return UNKNOWN;
}

uint64_t CGGIOpsData::get_bootstrap_resource_throughput() const {
  return get_canonical_throughput();
}

uint64_t CGGIOpsData::get_keyswitch_resource_fc() const { return 128; }

// 10% of cycles of PBS
uint64_t CGGIOpsData::get_keyswitch_resource_throughput() const {
  return static_cast<uint64_t>(0.10f * get_canonical_throughput());
}

cornami::OpCoreCountAndLatencyT CGGIOpsData::get_resource_use(
    CGGIOpsEnum opkind, uint64_t count) const {
  cornami::OpCoreCountAndLatencyT resource;
  // FC use
  resource.first =
      (m_bs_and_ks.at(opkind).at(0) ? get_bootstrap_resource_fc() * count : 0) +
      (m_bs_and_ks.at(opkind).at(1) ? get_keyswitch_resource_fc() * count : 0);
  // Throughput use
  resource.second = (m_bs_and_ks.at(opkind).at(0)
                         ? get_bootstrap_resource_throughput() * count
                         : 0) +
                    (m_bs_and_ks.at(opkind).at(1)
                         ? get_keyswitch_resource_throughput() * count
                         : 0);
  return resource;
}

}  // namespace heir

namespace cornami {
#define GEN_PASS_DEF_CGGIESTIMATOR
#include "lib/Backend/cornami/Dialect/SCIFRBool/Analysis/Passes.h.inc"

using namespace mlir::heir;

struct CGGIEstimator : impl::CGGIEstimatorBase<CGGIEstimator> {
  using CGGIEstimatorBase::CGGIEstimatorBase;
  PerfCounter m_data;

  void runOnOperation() override {
    // MLIRContext &context = getContext();
    graph::Graph<Operation*> graph;
    CGGIOpsData m_cggiops(2048, 1, 1, 841);
    auto m_cp_ccgiops = m_cggiops;

    getOperation()->walk<WalkOrder::PostOrder>([&](Operation* op) {
      const bool result = m_cggiops.analyze(op);
      if (!result) {
        op->emitWarning() << "Unknown operator not counted for Estimate\n";
        return WalkResult::advance();
      }

      // op is a strictly CGGI operator only
      graph.addVertex(op);
      if (op->getNumOperands() > 0) {
        llvm::outs() << "Operator : "
                     << op->getName().getIdentifier().getValue().str()
                     << " has " << op->getNumOperands() << " operands.\n";
        for (auto inoperand : op->getOperands()) {
          if (!inoperand.getDefiningOp()) continue;
          graph.addVertex(inoperand.getDefiningOp());
          int latency =
              (int)m_cggiops.get_resource_use(m_cggiops.get_op_kind(op), 1)
                  .second;
          graph.addEdge(inoperand.getDefiningOp(), op, latency);
        }
      }

      llvm::outs() << "operation = " << "\n";
      op->dump();
      llvm::outs() << "\n";
      return WalkResult::advance();
    });
    m_cggiops.display();

    dump(graph);
    // auto cp_ = graph.get_longest_source_to_sink_path();
    auto cp_ = graph.findApproximateCriticalPath();
    for (auto op : graph.getVertices()) {
      llvm::outs() << "Op = " << op->getName().getIdentifier().getValue().str()
                   << "\n";
    }
    if (succeeded(cp_) && cp_.value().size() > 0) {
      llvm::outs() << "============ Critical Path ============\n";
      auto cp = cp_.value();
      for (uint64_t idx = 0; idx < cp.size(); idx++) {
        auto op = cp[idx];
#if 0
        llvm::outs() << "Operation : " << idx << "\n\t";
        op->dump();
#endif
        m_cp_ccgiops.analyze(op);
      }
      m_cggiops.display();
      llvm::outs() << "=======================================\n";
    } else {
      llvm::outs() << "======= Cannot find useful CP =========\n";
    }
  }
};  // struct CGGIEstimator

}  // namespace cornami
}  // namespace mlir
