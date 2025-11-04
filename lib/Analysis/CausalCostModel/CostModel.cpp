#include "lib/Analysis/CausalCostModel/CostModel.h"

#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace causal {

SimpleCostModel::SimpleCostModel() {
  // Initialize with simple fixed costs based on operation type
  // These are rough estimates based on FHE operation costs

  // Addition: fast, no depth, minimal noise
  base_costs_["add"] = CostMetrics{0.1, 1024, 0, 0, 0.01};
  base_costs_["sub"] = CostMetrics{0.1, 1024, 0, 0, 0.01};

  // Multiplication: slow, increases depth, significant noise
  base_costs_["mul"] = CostMetrics{5.0, 10240, 1, 0, 2.0};
  base_costs_["mul_plain"] = CostMetrics{0.5, 5120, 0, 0, 0.5};

  // Rotation: medium speed, requires key switching
  base_costs_["rot"] = CostMetrics{2.0, 20480, 0, 1, 0.1};
  base_costs_["rotate"] = CostMetrics{2.0, 20480, 0, 1, 0.1};

  // Relinearization: expensive key switching
  base_costs_["relin"] = CostMetrics{3.0, 15360, 0, 0, 0.0};

  // Bootstrapping: very expensive
  base_costs_["bootstrap"] = CostMetrics{45.0, 102400, 0, 0, -10.0};
}

CostMetrics SimpleCostModel::computeCost(Operation* op,
                                          const CostContext& context) {
  std::string opName = op->getName().getStringRef().str();

  // Find base cost for this operation type
  CostMetrics cost;
  for (const auto& [key, value] : base_costs_) {
    if (opName.find(key) != std::string::npos) {
      cost = value;
      break;
    }
  }

  // Apply context-dependent adjustments
  if (context.on_critical_path) {
    cost.latency_ms *= 1.5;
  }

  if (context.depth_remaining < cost.depth_consumed) {
    // Need bootstrap
    cost = cost + base_costs_["bootstrap"];
  }

  return cost;
}

CostMetrics SimpleCostModel::computeKernelCost(
    const std::shared_ptr<void>& dag, const CostContext& context) {
  // For SimpleCostModel, just return a fixed cost
  // A real implementation would traverse the DAG
  return CostMetrics{10.0, 50000, 5, 10, 1.0};
}

}  // namespace causal
}  // namespace heir
}  // namespace mlir
