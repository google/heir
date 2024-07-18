#ifndef LIB_GRAPH_GRAPH_H_
#define LIB_GRAPH_GRAPH_H_

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace graph {

// A graph data structure.
//
// Parameter `V` is the vertex type, which is expected to be cheap to copy.
template <typename V>
class Graph {
 public:
  // Adds a vertex to the graph
  void addVertex(V vertex) { vertices.insert(vertex); }

  // Adds an edge from the given `source` to the given `target`. Returns false
  // if either the source or target is not a previously inserted vertex, and
  // returns true otherwise. The graph is unchanged if false is returned.
  bool addEdge(V source, V target) {
    if (vertices.count(source) == 0 || vertices.count(target) == 0) {
      return false;
    }
    outEdges[source].insert(target);
    inEdges[target].insert(source);
    return true;
  }

  // Returns true iff the given vertex has previously been added to the graph
  // using `AddVertex`.
  bool contains(V vertex) { return vertices.count(vertex) > 0; }

  bool empty() { return vertices.empty(); }

  const std::set<V>& getVertices() { return vertices; }

  // Returns the edges that point out of the given vertex.
  std::vector<V> edgesOutOf(V vertex) {
    if (vertices.count(vertex)) {
      std::vector<V> result(outEdges[vertex].begin(), outEdges[vertex].end());
      // Note: The vertices are sorted to ensure determinism in the output.
      std::sort(result.begin(), result.end());
      return result;
    }
    return {};
  }

  // Returns the edges that point into the given vertex.
  std::vector<V> edgesInto(V vertex) {
    if (vertices.count(vertex)) {
      std::vector<V> result(inEdges[vertex].begin(), inEdges[vertex].end());
      // Note: The vertices are sorted to ensure determinism in the output.
      std::sort(result.begin(), result.end());
      return result;
    }
    return {};
  }

  // Returns a topological sort of the nodes in the graph if the graph is
  // acyclic, otherwise returns failure()
  FailureOr<std::vector<V>> topologicalSort() {
    std::vector<V> result;

    // Kahn's algorithm
    std::vector<V> active;
    std::map<V, int64_t> edgeCount;
    for (const V& vertex : vertices) {
      edgeCount[vertex] = edgesInto(vertex).size();
      if (edgeCount.at(vertex) == 0) {
        active.push_back(vertex);
      }
    }

    while (!active.empty()) {
      V source = active.back();
      active.pop_back();
      result.push_back(source);
      for (auto target : edgesOutOf(source)) {
        edgeCount[target]--;
        if (edgeCount.at(target) == 0) {
          active.push_back(target);
        }
      }
    }

    if (result.size() != vertices.size()) {
      return failure();
    }

    return result;
  }

  // Find the level of each node in the graph, where the level
  // is the length of the longest path from any input node to that node.
  //
  // Note: this algorithm doesn't optimize for the most "balanced" levels.
  // Algorithms that result in better balancing of nodes across levels include
  // the Coffman-Graham algorithm.
  //
  // Possible improvements we could make:
  //
  //  - Add a width restriction so that a level has bounded size.
  //  - Add a compabibility restriction for what operations can be in the same
  //    level.
  //
  FailureOr<std::vector<std::vector<V>>> sortGraphByLevels() {
    // Topologically sort the adjacency graph, then reverse it.
    auto result = topologicalSort();
    if (failed(result)) {
      return failure();
    }
    auto topoOrder = result.value();
    std::reverse(topoOrder.begin(), topoOrder.end());
    std::map<V, int> levels;

    // Assign levels to the nodes:
    // Traverse through the reversed topologically sorted nodes
    // (working backwards through the graph, starting from the outputs)
    // and assign the level of each node as 1 + the maximum of all the
    // destinations of that node and -1, such that the first node processed
    // (an output node) will have level = 0.
    int maxLevel = 0;
    int maxSourceLevel = -1;
    for (auto vertex : topoOrder) {
      maxSourceLevel = -1;
      for (auto edge : edgesOutOf(vertex)) {
        maxSourceLevel = std::max(maxSourceLevel, levels[edge]);
      }
      levels[vertex] = 1 + maxSourceLevel;
      maxLevel = std::max(levels.at(vertex), maxLevel);
    }

    // Output will be a vector of vectors of the nodes at each level.
    // Reverse the levels values, such that input nodes have smaller level
    // values.
    std::vector<std::vector<V>> output(maxLevel + 1);
    for (auto entry : levels) {
      output[maxLevel - entry.second].push_back(entry.first);
    }
    return output;
  }

 private:
  std::set<V> vertices;
  std::map<V, std::set<V>> outEdges;
  std::map<V, std::set<V>> inEdges;
};

}  // namespace graph
}  // namespace heir
}  // namespace mlir

#endif  // LIB_GRAPH_GRAPH_H_
