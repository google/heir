#ifndef LIB_ANALYSIS_CAUSALCOSTMODEL_COSTMODEL_H_
#define LIB_ANALYSIS_CAUSALCOSTMODEL_COSTMODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace causal {

/// Multi-dimensional cost metrics for FHE operations
struct CostMetrics {
  double latency_ms = 0.0;        // Execution time in milliseconds
  size_t memory_bytes = 0;        // Memory overhead in bytes
  int64_t depth_consumed = 0;     // Multiplicative depth used
  int64_t rotations = 0;          // Number of rotations
  double noise_growth = 0.0;      // Estimated noise increase

  CostMetrics() = default;
  CostMetrics(double lat, size_t mem, int64_t depth, int64_t rot, double noise)
      : latency_ms(lat),
        memory_bytes(mem),
        depth_consumed(depth),
        rotations(rot),
        noise_growth(noise) {}

  /// Combine two cost metrics (for sequential operations)
  CostMetrics operator+(const CostMetrics& other) const {
    return CostMetrics{latency_ms + other.latency_ms,
                       memory_bytes + other.memory_bytes,
                       depth_consumed + other.depth_consumed,
                       rotations + other.rotations,
                       noise_growth + other.noise_growth};
  }

  /// Scale cost metrics by a factor
  CostMetrics operator*(double factor) const {
    return CostMetrics{latency_ms * factor,
                       static_cast<size_t>(memory_bytes * factor),
                       static_cast<int64_t>(depth_consumed * factor),
                       static_cast<int64_t>(rotations * factor),
                       noise_growth * factor};
  }

  /// Check if this cost dominates another (Pareto dominance)
  bool dominates(const CostMetrics& other) const {
    return latency_ms <= other.latency_ms &&
           memory_bytes <= other.memory_bytes &&
           depth_consumed <= other.depth_consumed &&
           rotations <= other.rotations &&
           noise_growth <= other.noise_growth &&
           (latency_ms < other.latency_ms ||
            memory_bytes < other.memory_bytes ||
            depth_consumed < other.depth_consumed ||
            rotations < other.rotations ||
            noise_growth < other.noise_growth);
  }

  /// Compute weighted objective value
  double weightedObjective(double w_latency, double w_memory, double w_depth,
                           double w_rotations) const {
    return w_latency * latency_ms + w_memory * memory_bytes +
           w_depth * depth_consumed + w_rotations * rotations;
  }
};

/// Context information for cost computation
struct CostContext {
  bool on_critical_path = false;      // Whether op is on critical path
  bool parallelizable = false;        // Whether can run in parallel
  int64_t depth_remaining = 100;      // Remaining depth budget
  size_t memory_available = 1024*1024*1024;  // Available memory
  std::string backend = "OpenFHE";    // Target backend
  std::string scheme = "CKKS";        // FHE scheme
  int64_t ring_dimension = 16384;     // Security parameter
  int64_t thread_count = 1;           // Available threads

  CostContext() = default;
};

/// Abstract interface for cost models
/// Allows swapping different cost model implementations
class CostModel {
 public:
  virtual ~CostModel() = default;

  /// Compute cost of an operation in a given context
  virtual CostMetrics computeCost(Operation* op,
                                   const CostContext& context) = 0;

  /// Compute cost of a kernel implementation (DAG)
  virtual CostMetrics computeKernelCost(
      const std::shared_ptr<void>& dag,  // ArithmeticDagNode
      const CostContext& context) = 0;

  /// Get the name of this cost model
  virtual std::string getName() const = 0;

  /// Initialize the cost model with configuration
  virtual void initialize(const std::unordered_map<std::string, std::string>&
                              config) {}
};

/// Simple cost model that uses fixed costs per operation type
class SimpleCostModel : public CostModel {
 public:
  SimpleCostModel();

  CostMetrics computeCost(Operation* op,
                          const CostContext& context) override;

  CostMetrics computeKernelCost(const std::shared_ptr<void>& dag,
                                 const CostContext& context) override;

  std::string getName() const override { return "SimpleCostModel"; }

 private:
  std::unordered_map<std::string, CostMetrics> base_costs_;
};

}  // namespace causal
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_CAUSALCOSTMODEL_COSTMODEL_H_
