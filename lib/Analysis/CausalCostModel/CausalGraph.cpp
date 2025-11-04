#include "lib/Analysis/CausalCostModel/CausalGraph.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <unordered_set>

namespace mlir {
namespace heir {
namespace causal {

CausalGraph::CausalGraph() {}

void CausalGraph::buildFHECausalStructure() {
  // Clear existing structure
  nodes_.clear();
  edges_.clear();
  node_map_.clear();

  // ========== EXOGENOUS VARIABLES (CAUSES) ==========

  // Operation properties
  addNode(NodeType::OPERATION_TYPE, "op_type");
  addNode(NodeType::OPERAND_TYPES, "operand_types");
  addNode(NodeType::ROTATION_OFFSET, "rotation_offset");

  // Backend configuration (confounders - affect everything)
  addNode(NodeType::BACKEND, "backend");
  addNode(NodeType::SCHEME, "scheme");
  addNode(NodeType::SECURITY_PARAMS, "security_params");
  addNode(NodeType::HARDWARE_CONFIG, "hardware_config");

  // ========== INTERMEDIATE EFFECTS (MEDIATORS) ==========

  // Primary effects of operations
  addNode(NodeType::NOISE_GROWTH, "noise_growth");
  addNode(NodeType::DEPTH_CONSUMED, "depth_consumed");
  addNode(NodeType::KEY_SWITCHING_REQUIRED, "key_switching");
  addNode(NodeType::RELINEARIZATION_REQUIRED, "relinearization");

  // Graph context variables
  addNode(NodeType::ON_CRITICAL_PATH, "critical_path");
  addNode(NodeType::PARALLELIZABLE, "parallelizable");
  addNode(NodeType::DEPTH_REMAINING, "depth_remaining");
  addNode(NodeType::MEMORY_PRESSURE, "memory_pressure");

  // ========== OBSERVABLE OUTCOMES ==========

  addNode(NodeType::LATENCY, "latency");
  addNode(NodeType::MEMORY, "memory");
  addNode(NodeType::NOISE_LEVEL, "noise_level");
  addNode(NodeType::THROUGHPUT, "throughput");

  // ========== CAUSAL EDGES (DIRECT EFFECTS) ==========

  // Multiplication causes quadratic noise growth and depth consumption
  // mul → noise_growth (strong, multiplicative)
  addEdge("op_type", "noise_growth", 2.0);  // quadratic for mul
  addEdge("op_type", "depth_consumed", 1.0);
  addEdge("op_type", "relinearization", 1.0);  // mul requires relin

  // Rotation requires key switching
  addEdge("rotation_offset", "key_switching", 1.0);
  addEdge("op_type", "key_switching", 1.0);  // rotation type

  // Addition causes linear noise growth
  addEdge("op_type", "noise_growth", 0.1);  // additive for add

  // Noise growth affects final noise level
  addEdge("noise_growth", "noise_level", 1.0);

  // Key switching and relinearization affect latency
  addEdge("key_switching", "latency", 10.0);  // expensive operation
  addEdge("relinearization", "latency", 8.0);  // expensive operation

  // Depth consumption affects whether bootstrapping needed
  addEdge("depth_consumed", "depth_remaining", -1.0);
  addEdge("depth_remaining", "latency", -5.0);  // low depth → bootstrap

  // Memory effects
  addEdge("key_switching", "memory", 5.0);  // rotation keys
  addEdge("relinearization", "memory", 3.0);  // relin keys

  // Context-dependent effects
  addEdge("critical_path", "latency", 2.0);  // can't parallelize
  addEdge("parallelizable", "throughput", 3.0);  // high parallelism
  addEdge("memory_pressure", "latency", 1.5);  // cache misses

  // ========== CONFOUNDING PATHS ==========

  // Backend affects everything
  auto* backendEdge1 = addEdge("backend", "latency", 5.0);
  backendEdge1->setDirectCausal(false);  // confounding path

  auto* backendEdge2 = addEdge("backend", "noise_growth", 1.0);
  backendEdge2->setDirectCausal(false);

  auto* backendEdge3 = addEdge("backend", "key_switching", 2.0);
  backendEdge3->setDirectCausal(false);

  // Scheme affects noise behavior
  auto* schemeEdge1 = addEdge("scheme", "noise_growth", 1.5);
  schemeEdge1->setDirectCausal(false);

  auto* schemeEdge2 = addEdge("scheme", "depth_consumed", 1.0);
  schemeEdge2->setDirectCausal(false);

  // Security parameters affect everything
  addEdge("security_params", "latency", 3.0);
  addEdge("security_params", "memory", 4.0);
  addEdge("security_params", "noise_level", 2.0);

  // Hardware config affects parallelism and latency
  addEdge("hardware_config", "parallelizable", 2.0);
  addEdge("hardware_config", "latency", -1.0);  // more threads = faster

  // Operand types affect costs (ct-ct vs ct-pt)
  addEdge("operand_types", "noise_growth", 0.5);
  addEdge("operand_types", "latency", 0.8);
}

CausalNode* CausalGraph::addNode(NodeType type, const std::string& name) {
  auto node = std::make_unique<CausalNode>(type, name);
  CausalNode* nodePtr = node.get();
  nodes_.push_back(std::move(node));
  node_map_[name] = nodePtr;
  return nodePtr;
}

CausalEdge* CausalGraph::addEdge(const std::string& from, const std::string& to,
                                   double weight) {
  auto* fromNode = getNode(from);
  auto* toNode = getNode(to);

  if (!fromNode || !toNode) {
    return nullptr;
  }

  auto edge = std::make_unique<CausalEdge>(fromNode, toNode, weight);
  CausalEdge* edgePtr = edge.get();
  edges_.push_back(std::move(edge));

  // Update adjacency
  fromNode->addChild(toNode);
  toNode->addParent(fromNode);

  return edgePtr;
}

CausalNode* CausalGraph::getNode(const std::string& name) {
  auto it = node_map_.find(name);
  return (it != node_map_.end()) ? it->second : nullptr;
}

std::unordered_map<std::string, double> CausalGraph::computeEffects(
    const std::unordered_map<std::string, double>& interventions,
    const std::vector<std::string>& outcomes) {

  // Step 1: Set intervention values (do-operator)
  for (const auto& [nodeName, value] : interventions) {
    auto* node = getNode(nodeName);
    if (node) {
      node->setValue(value);
    }
  }

  // Step 2: Topological sort for causal ordering
  auto sorted = topologicalSort();

  // Step 3: Propagate values through the graph
  for (auto* node : sorted) {
    // Skip if this node was set by intervention
    if (interventions.count(node->getName()) > 0) {
      continue;
    }

    // Compute value as weighted sum of parent values
    double value = 0.0;
    for (auto* parent : node->getParents()) {
      // Find edge weight
      for (const auto& edge : edges_) {
        if (edge->getFrom() == parent && edge->getTo() == node) {
          value += parent->getValue() * edge->getWeight();
          break;
        }
      }
    }
    node->setValue(value);
  }

  // Step 4: Extract outcome values
  std::unordered_map<std::string, double> results;
  for (const auto& outcomeName : outcomes) {
    auto* node = getNode(outcomeName);
    if (node) {
      results[outcomeName] = node->getValue();
    }
  }

  return results;
}

bool CausalGraph::dSeparated(const std::string& x, const std::string& y,
                               const std::vector<std::string>& conditioning_set) {
  // Implement d-separation using Bayes-ball algorithm
  // For now, simplified implementation

  auto* xNode = getNode(x);
  auto* yNode = getNode(y);
  if (!xNode || !yNode) return true;

  // Convert conditioning set to node set
  std::unordered_set<CausalNode*> conditioned;
  for (const auto& name : conditioning_set) {
    auto* node = getNode(name);
    if (node) conditioned.insert(node);
  }

  // BFS to check if there's an unblocked path from x to y
  std::queue<CausalNode*> queue;
  std::unordered_set<CausalNode*> visited;

  queue.push(xNode);
  visited.insert(xNode);

  while (!queue.empty()) {
    auto* current = queue.front();
    queue.pop();

    if (current == yNode) {
      return false;  // Found unblocked path
    }

    // Check children (unless blocked by conditioning)
    if (conditioned.count(current) == 0) {
      for (auto* child : current->getChildren()) {
        if (visited.count(child) == 0) {
          visited.insert(child);
          queue.push(child);
        }
      }
    }
  }

  return true;  // No unblocked path found, so d-separated
}

std::vector<CausalNode*> CausalGraph::topologicalSort() {
  std::vector<CausalNode*> sorted;
  std::unordered_set<CausalNode*> visited;
  std::unordered_set<CausalNode*> inProgress;

  std::function<void(CausalNode*)> visit = [&](CausalNode* node) {
    if (visited.count(node)) return;
    if (inProgress.count(node)) {
      // Cycle detected - shouldn't happen in a DAG
      return;
    }

    inProgress.insert(node);
    for (auto* parent : node->getParents()) {
      visit(parent);
    }
    inProgress.erase(node);
    visited.insert(node);
    sorted.push_back(node);
  };

  for (const auto& node : nodes_) {
    visit(node.get());
  }

  return sorted;
}

void CausalGraph::propagateValues() {
  auto sorted = topologicalSort();
  for (auto* node : sorted) {
    // Value already set by intervention or computed
    continue;
  }
}

}  // namespace causal
}  // namespace heir
}  // namespace mlir
