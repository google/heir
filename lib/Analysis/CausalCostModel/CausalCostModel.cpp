#include "lib/Analysis/CausalCostModel/CausalCostModel.h"

#include <algorithm>
#include <cmath>

namespace mlir {
namespace heir {
namespace causal {

std::unordered_map<std::string, double> FeatureVector::toCausalVariables()
    const {
  std::unordered_map<std::string, double> variables;

  // Operation type encoding (simplified)
  if (op_type.find("mul") != std::string::npos) {
    variables["op_type"] = 2.0;  // multiply
  } else if (op_type.find("add") != std::string::npos ||
             op_type.find("sub") != std::string::npos) {
    variables["op_type"] = 1.0;  // add/sub
  } else if (op_type.find("rot") != std::string::npos) {
    variables["op_type"] = 3.0;  // rotation
  } else {
    variables["op_type"] = 0.0;  // other
  }

  // Rotation offset
  variables["rotation_offset"] = static_cast<double>(rotation_offset);

  // Operand types
  if (operand_types == "ct-ct") {
    variables["operand_types"] = 2.0;  // ciphertext-ciphertext
  } else if (operand_types == "ct-pt") {
    variables["operand_types"] = 1.0;  // ciphertext-plaintext
  } else {
    variables["operand_types"] = 0.0;  // other
  }

  // Backend encoding
  if (backend == "OpenFHE") {
    variables["backend"] = 1.0;
  } else if (backend == "Lattigo") {
    variables["backend"] = 2.0;
  } else if (backend == "SEAL") {
    variables["backend"] = 3.0;
  } else {
    variables["backend"] = 0.0;
  }

  // Scheme encoding
  if (scheme == "CKKS") {
    variables["scheme"] = 1.0;
  } else if (scheme == "BGV") {
    variables["scheme"] = 2.0;
  } else if (scheme == "BFV") {
    variables["scheme"] = 3.0;
  } else {
    variables["scheme"] = 0.0;
  }

  // Security parameters
  variables["security_params"] = static_cast<double>(ring_dimension) / 16384.0;
  variables["hardware_config"] = static_cast<double>(thread_count);

  // Context variables
  variables["critical_path"] = on_critical_path ? 1.0 : 0.0;
  variables["parallelizable"] = parallelizable ? 1.0 : 0.0;
  variables["depth_remaining"] = static_cast<double>(depth_remaining);
  variables["memory_pressure"] =
      (memory_available > 0) ? (1.0 - std::min(1.0, memory_available / 1e9))
                             : 0.0;

  return variables;
}

CausalCostModel::CausalCostModel() {
  // Build the FHE causal structure
  causal_graph_.buildFHECausalStructure();
}

void CausalCostModel::initialize(
    const std::unordered_map<std::string, std::string>& config) {
  // Configuration could specify backend, scheme, etc.
  // For now, keep it simple
}

CostMetrics CausalCostModel::computeCost(Operation* op,
                                          const CostContext& context) {
  // Extract features from the operation
  FeatureVector features = extractFeatures(op, context);

  // Compute base cost using causal inference
  CostMetrics base = computeBaseCost(features);

  // Adjust for context
  return adjustForContext(base, features);
}

CostMetrics CausalCostModel::computeKernelCost(
    const std::shared_ptr<void>& dag_ptr, const CostContext& context) {
  if (!dag_ptr) {
    return CostMetrics{};
  }

  // Cast to proper DAG type
  auto dag = std::static_pointer_cast<
      kernel::ArithmeticDagNode<kernel::SymbolicValue>>(dag_ptr);

  // Extract features from the DAG
  FeatureVector features = extractFeaturesFromDAG(dag, context);

  // Compute base cost using causal inference
  CostMetrics base = computeBaseCost(features);

  // Adjust for context
  return adjustForContext(base, features);
}

void CausalCostModel::addBenchmarkSamples(
    const std::vector<BenchmarkSample>& samples) {
  benchmark_samples_.insert(benchmark_samples_.end(), samples.begin(),
                            samples.end());
}

void CausalCostModel::learnFromBenchmarks() {
  if (benchmark_samples_.empty()) {
    // No data to learn from, use default weights
    weights_learned_ = false;
    return;
  }

  // Simple learning: weighted averaging of causal effects
  // In a full implementation, this would use regression or other ML

  std::unordered_map<std::string, double> sum_effects;
  std::unordered_map<std::string, double> total_weights;

  for (const auto& sample : benchmark_samples_) {
    double weight = sample.weight;
    auto variables = sample.features.toCausalVariables();

    // Accumulate weighted effects
    for (const auto& [var, value] : variables) {
      sum_effects[var] += value * weight * sample.observed_cost.latency_ms;
      total_weights[var] += weight;
    }
  }

  // Compute weighted average effects
  for (const auto& [var, sum] : sum_effects) {
    if (total_weights[var] > 0) {
      learned_weights_[var] = sum / total_weights[var];
    }
  }

  weights_learned_ = true;
}

FeatureVector CausalCostModel::extractFeatures(Operation* op,
                                                 const CostContext& context) {
  FeatureVector features;

  // Extract operation type
  features.op_type = op->getName().getStringRef().str();

  // Extract operand types (simplified)
  if (op->getNumOperands() >= 2) {
    features.operand_types = "ct-ct";  // Assume both ciphertext
  } else if (op->getNumOperands() == 1) {
    features.operand_types = "ct-pt";  // Assume one ciphertext
  }

  // Extract rotation offset if it's a rotation operation
  if (features.op_type.find("rot") != std::string::npos) {
    // Try to extract offset from attributes
    if (auto offsetAttr = op->getAttrOfType<IntegerAttr>("offset")) {
      features.rotation_offset = offsetAttr.getInt();
    } else if (auto indexAttr = op->getAttrOfType<IntegerAttr>("index")) {
      features.rotation_offset = indexAttr.getInt();
    }
  }

  // Copy context features
  features.on_critical_path = context.on_critical_path;
  features.parallelizable = context.parallelizable;
  features.depth_remaining = context.depth_remaining;
  features.memory_available = context.memory_available;
  features.backend = context.backend;
  features.scheme = context.scheme;
  features.ring_dimension = context.ring_dimension;
  features.thread_count = context.thread_count;

  return features;
}

FeatureVector CausalCostModel::extractFeaturesFromDAG(
    const std::shared_ptr<kernel::ArithmeticDagNode<kernel::SymbolicValue>>&
        dag,
    const CostContext& context) {
  FeatureVector features;

  // Use our visitors to extract DAG properties
  kernel::RotationCountVisitor rotCounter;
  kernel::MultiplicativeDepthVisitor depthCounter;

  features.rotation_offset = rotCounter.process(dag);
  int64_t depth = depthCounter.process(dag);

  // Set operation type based on DAG complexity
  if (depth > 0) {
    features.op_type = "kernel_with_multiply";
  } else if (features.rotation_offset > 0) {
    features.op_type = "kernel_with_rotation";
  } else {
    features.op_type = "kernel_simple";
  }

  features.operand_types = "ct-ct";  // Kernels typically operate on ciphertexts

  // Copy context features
  features.on_critical_path = context.on_critical_path;
  features.parallelizable = context.parallelizable;
  features.depth_remaining = context.depth_remaining;
  features.memory_available = context.memory_available;
  features.backend = context.backend;
  features.scheme = context.scheme;
  features.ring_dimension = context.ring_dimension;
  features.thread_count = context.thread_count;

  return features;
}

CostMetrics CausalCostModel::computeBaseCost(const FeatureVector& features) {
  // Convert features to causal variables
  auto variables = features.toCausalVariables();

  // Perform causal inference
  auto effects = causal_graph_.computeEffects(
      variables, {"latency", "memory", "noise_level", "throughput"});

  // Extract cost metrics from causal effects
  CostMetrics cost;
  cost.latency_ms = effects["latency"];
  cost.memory_bytes = static_cast<size_t>(effects["memory"] * 1024 * 1024);
  cost.noise_growth = effects["noise_level"];

  // Use simple estimates for depth and rotations from features
  cost.depth_consumed = (features.op_type.find("mul") != std::string::npos) ? 1 : 0;
  cost.rotations = (features.rotation_offset != 0) ? 1 : 0;

  // Apply learned weights if available
  if (weights_learned_) {
    double learned_latency = 0.0;
    for (const auto& [var, value] : variables) {
      if (learned_weights_.count(var)) {
        learned_latency += value * learned_weights_[var];
      }
    }
    // Blend causal model with learned model
    cost.latency_ms = 0.7 * cost.latency_ms + 0.3 * learned_latency;
  }

  return cost;
}

CostMetrics CausalCostModel::adjustForContext(const CostMetrics& base,
                                                const FeatureVector& features) {
  CostMetrics adjusted = base;

  // Critical path penalty (can't parallelize)
  if (features.on_critical_path) {
    adjusted.latency_ms *= 1.5;  // 50% penalty
  }

  // Parallelism discount
  if (features.parallelizable && features.thread_count > 1) {
    double parallelism_factor =
        1.0 / std::min(4.0, static_cast<double>(features.thread_count));
    adjusted.latency_ms *= parallelism_factor;
  }

  // Depth budget exhausted requires bootstrapping
  if (features.depth_remaining < adjusted.depth_consumed) {
    // CKKS bootstrap ~40-50ms per slot
    adjusted.latency_ms += 45.0;
    adjusted.memory_bytes += 100 * 1024 * 1024;  // 100 MB for bootstrap keys
  }

  // Memory pressure causes slowdown (cache misses)
  if (features.memory_available < adjusted.memory_bytes) {
    adjusted.latency_ms *= 1.3;  // 30% slowdown from memory pressure
  }

  return adjusted;
}

double CausalCostModel::computeImportanceWeight(
    const BenchmarkSample& sample, const CostContext& target_context) {
  // Compute similarity between sample and target context
  double backend_similarity =
      (sample.features.backend == target_context.backend) ? 1.0 : 0.3;
  double scheme_similarity =
      (sample.features.scheme == target_context.scheme) ? 1.0 : 0.5;

  // Recency weight (more recent samples weighted higher)
  // Simplified: assume samples are already sorted by timestamp
  double recency_weight = 1.0;  // Could decay exponentially

  return backend_similarity * scheme_similarity * recency_weight;
}

}  // namespace causal
}  // namespace heir
}  // namespace mlir
