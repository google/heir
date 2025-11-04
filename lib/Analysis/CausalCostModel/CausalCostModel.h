#ifndef LIB_ANALYSIS_CAUSALCOSTMODEL_CAUSALCOSTMODEL_H_
#define LIB_ANALYSIS_CAUSALCOSTMODEL_CAUSALCOSTMODEL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lib/Analysis/CausalCostModel/CausalGraph.h"
#include "lib/Analysis/CausalCostModel/CostModel.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/MultiplicativeDepthVisitor.h"
#include "lib/Kernel/RotationCountVisitor.h"
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace causal {

/// Feature vector extracted from an operation and its context
struct FeatureVector {
  // Operation features
  std::string op_type;
  std::string operand_types;  // "ct-ct", "ct-pt", etc.
  int64_t rotation_offset = 0;

  // Context features
  bool on_critical_path = false;
  bool parallelizable = false;
  int64_t depth_remaining = 0;
  size_t memory_available = 0;

  // Backend features
  std::string backend;
  std::string scheme;
  int64_t ring_dimension = 0;
  int64_t thread_count = 1;

  /// Convert features to causal graph variable assignments
  std::unordered_map<std::string, double> toCausalVariables() const;
};

/// Benchmark sample for learning cost models
struct BenchmarkSample {
  FeatureVector features;
  CostMetrics observed_cost;
  double weight = 1.0;  // Importance weight for learning
  int64_t timestamp = 0;  // For recency weighting
};

/// Causal cost model that uses causal inference
class CausalCostModel : public CostModel {
 public:
  CausalCostModel();

  CostMetrics computeCost(Operation* op,
                          const CostContext& context) override;

  CostMetrics computeKernelCost(const std::shared_ptr<void>& dag,
                                 const CostContext& context) override;

  std::string getName() const override { return "CausalCostModel"; }

  void initialize(const std::unordered_map<std::string, std::string>& config)
      override;

  /// Add benchmark samples for learning
  void addBenchmarkSamples(const std::vector<BenchmarkSample>& samples);

  /// Learn edge weights from benchmark data
  void learnFromBenchmarks();

  /// Get the underlying causal graph
  CausalGraph* getCausalGraph() { return &causal_graph_; }

 private:
  CausalGraph causal_graph_;
  std::vector<BenchmarkSample> benchmark_samples_;

  // Learned parameters
  std::unordered_map<std::string, double> learned_weights_;
  bool weights_learned_ = false;

  /// Extract features from an operation
  FeatureVector extractFeatures(Operation* op, const CostContext& context);

  /// Extract features from a DAG
  FeatureVector extractFeaturesFromDAG(
      const std::shared_ptr<kernel::ArithmeticDagNode<kernel::SymbolicValue>>&
          dag,
      const CostContext& context);

  /// Compute base cost using causal inference
  CostMetrics computeBaseCost(const FeatureVector& features);

  /// Adjust cost for context (critical path, parallelism, etc.)
  CostMetrics adjustForContext(const CostMetrics& base,
                                const FeatureVector& features);

  /// Compute importance weight for a benchmark sample
  double computeImportanceWeight(const BenchmarkSample& sample,
                                  const CostContext& target_context);
};

}  // namespace causal
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_CAUSALCOSTMODEL_CAUSALCOSTMODEL_H_
