#ifndef LIB_ANALYSIS_CAUSALCOSTMODEL_CAUSALGRAPH_H_
#define LIB_ANALYSIS_CAUSALCOSTMODEL_CAUSALGRAPH_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir {
namespace heir {
namespace causal {

// Forward declarations
class CausalNode;
class CausalEdge;

/// Represents different types of causal variables in the FHE cost model
enum class NodeType {
  // Operation properties (exogenous variables)
  OPERATION_TYPE,      // add, mul, rotate, relin, etc.
  OPERAND_TYPES,       // ct-ct, ct-pt, etc.
  ROTATION_OFFSET,     // rotation distance

  // Backend configuration (confounders)
  BACKEND,             // OpenFHE, Lattigo, SEAL
  SCHEME,              // CKKS, BGV, BFV
  SECURITY_PARAMS,     // ring dimension, modulus chain
  HARDWARE_CONFIG,     // CPU/GPU, thread count

  // Intermediate effects (mediators)
  NOISE_GROWTH,        // how much noise increases
  DEPTH_CONSUMED,      // multiplicative depth used
  KEY_SWITCHING_REQUIRED,  // whether key switch needed
  RELINEARIZATION_REQUIRED,  // whether relin needed

  // Graph context (effect modifiers)
  ON_CRITICAL_PATH,    // whether operation is on critical path
  PARALLELIZABLE,      // whether can run in parallel
  DEPTH_REMAINING,     // remaining depth budget
  MEMORY_PRESSURE,     // current memory usage

  // Observable costs (outcome variables)
  LATENCY,             // execution time
  MEMORY,              // memory overhead
  NOISE_LEVEL,         // final noise level
  THROUGHPUT           // operations per second
};

/// Represents a node in the causal DAG
class CausalNode {
 public:
  CausalNode(NodeType type, const std::string& name)
      : type_(type), name_(name) {}

  NodeType getType() const { return type_; }
  const std::string& getName() const { return name_; }

  // Get/set current value (for inference)
  double getValue() const { return value_; }
  void setValue(double value) { value_ = value; }

  // Parents and children in the DAG
  const std::vector<CausalNode*>& getParents() const { return parents_; }
  const std::vector<CausalNode*>& getChildren() const { return children_; }

  void addParent(CausalNode* parent) { parents_.push_back(parent); }
  void addChild(CausalNode* child) { children_.push_back(child); }

 private:
  NodeType type_;
  std::string name_;
  double value_ = 0.0;
  std::vector<CausalNode*> parents_;   // Causes of this node
  std::vector<CausalNode*> children_;  // Effects of this node
};

/// Represents a causal edge with learned weight
class CausalEdge {
 public:
  CausalEdge(CausalNode* from, CausalNode* to, double weight = 0.0)
      : from_(from), to_(to), weight_(weight) {}

  CausalNode* getFrom() const { return from_; }
  CausalNode* getTo() const { return to_; }

  double getWeight() const { return weight_; }
  void setWeight(double weight) { weight_ = weight; }

  // Edge type: direct causal effect vs confounding path
  bool isDirectCausal() const { return is_direct_causal_; }
  void setDirectCausal(bool direct) { is_direct_causal_ = direct; }

 private:
  CausalNode* from_;
  CausalNode* to_;
  double weight_;  // Learned causal effect size
  bool is_direct_causal_ = true;  // vs confounding path
};

/// The causal graph structure for FHE cost modeling
class CausalGraph {
 public:
  CausalGraph();

  /// Build the FHE-specific causal structure
  void buildFHECausalStructure();

  /// Add a node to the graph
  CausalNode* addNode(NodeType type, const std::string& name);

  /// Add a causal edge
  CausalEdge* addEdge(const std::string& from, const std::string& to,
                      double weight = 0.0);

  /// Get a node by name
  CausalNode* getNode(const std::string& name);

  /// Perform causal inference: compute effects given interventions
  /// This implements do-calculus to compute P(outcome | do(intervention))
  std::unordered_map<std::string, double> computeEffects(
      const std::unordered_map<std::string, double>& interventions,
      const std::vector<std::string>& outcomes);

  /// Perform d-separation test to check conditional independence
  /// Used to identify confounders vs mediators
  bool dSeparated(const std::string& x, const std::string& y,
                  const std::vector<std::string>& conditioning_set);

  /// Get all nodes
  const std::vector<std::unique_ptr<CausalNode>>& getNodes() const {
    return nodes_;
  }

  /// Get all edges
  const std::vector<std::unique_ptr<CausalEdge>>& getEdges() const {
    return edges_;
  }

 private:
  std::vector<std::unique_ptr<CausalNode>> nodes_;
  std::vector<std::unique_ptr<CausalEdge>> edges_;
  std::unordered_map<std::string, CausalNode*> node_map_;

  /// Helper: topological sort for causal inference
  std::vector<CausalNode*> topologicalSort();

  /// Helper: propagate values through the graph
  void propagateValues();
};

}  // namespace causal
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_CAUSALCOSTMODEL_CAUSALGRAPH_H_
