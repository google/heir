#ifndef LIB_UTILS_GRAPH_GRAPH_H_
#define LIB_UTILS_GRAPH_GRAPH_H_

#include <algorithm>
#include <cstdint>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
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
  bool contains(V vertex) const { return vertices.count(vertex) > 0; }

  bool empty() const { return vertices.empty(); }

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

// An undirected graph data structure.
//
// Parameter `V` is the vertex type, which is expected to be cheap to copy.
template <typename V>
class UndirectedGraph {
 public:
  // Adds a vertex to the graph
  void addVertex(V vertex) { vertices.insert(vertex); }

  // Adds an edge between `source` and `target`. Returns false if either the
  // source or target is not a previously inserted vertex, and returns true
  // otherwise. The graph is unchanged if false is returned.
  bool addEdge(V source, V target) {
    if (vertices.count(source) == 0 || vertices.count(target) == 0) {
      return false;
    }
    edges[source].insert(target);
    edges[target].insert(source);
    return true;
  }

  // Returns true iff the given vertex has previously been added to the graph
  // using `AddVertex`.
  bool contains(V vertex) const { return vertices.count(vertex) > 0; }

  bool empty() const { return vertices.empty(); }

  const std::set<V>& getVertices() const { return vertices; }

  // Returns the edges incident to the given vertex.
  std::vector<V> edgesIncidentTo(V vertex) const {
    if (vertices.count(vertex) && edges.count(vertex)) {
      std::vector<V> result(edges.at(vertex).begin(), edges.at(vertex).end());
      // Note: The vertices are sorted to ensure determinism in the output.
      std::sort(result.begin(), result.end());
      return result;
    }
    return {};
  }

 private:
  std::set<V> vertices;
  std::map<V, std::set<V>> edges;
};

/// The greedy "DSatur" graph coloring algorithm
/// Cf. https://en.wikipedia.org/wiki/DSatur
template <typename V>
class GreedyGraphColoring {
 public:
  GreedyGraphColoring() = default;
  ~GreedyGraphColoring() = default;

  std::unordered_map<V, int> color(const UndirectedGraph<V>& graph) {
    colors.clear();
    neighborColors.clear();
    vertexSaturations.clear();

    for (const auto& vertex : graph.getVertices()) {
      neighborColors[vertex] = std::unordered_set<int>();
      updateSaturationDegree(graph, vertex);
    }

    while (!queue.empty()) {
      auto current = queue.top();
      queue.pop();

      // Skip if vertex is already colored or info is outdated. Could avoid
      // having "stale" info by using a std::set instead of a priority queue,
      // but then that would incur log(n) lookups and log(n) updates. Probably
      // the extra memory usage is fine since the graphs should be relatively
      // sparse.
      if (colors.find(current.vertex) != colors.end() ||
          current.saturationDegree !=
              vertexSaturations[current.vertex].saturationDegree ||
          current.uncoloredDegree !=
              vertexSaturations[current.vertex].uncoloredDegree) {
        continue;
      }

      // Use the smallest unused color among neighbors.
      int color = 0;
      while (neighborColors[current.vertex].find(color) !=
             neighborColors[current.vertex].end()) {
        color++;
      }

      colors[current.vertex] = color;
      updateNeighborSaturation(graph, current.vertex, color);
    }

    return colors;
  }

 private:
  struct VertexInfo {
    V vertex;
    // The number of different colors used by neighbors, primary sort key.
    int saturationDegree;
    // The number of uncolored neighbors, secondary sort key.
    int uncoloredDegree;

    bool operator<(const VertexInfo& other) const {
      if (saturationDegree != other.saturationDegree)
        return saturationDegree < other.saturationDegree;
      if (uncoloredDegree != other.uncoloredDegree)
        return uncoloredDegree < other.uncoloredDegree;
      // Visit smaller index vertices first in a tiebreak
      return vertex > other.vertex;
    }
  };

  void updateSaturationDegree(const UndirectedGraph<V>& graph,
                              const V& vertex) {
    auto neighbors = graph.edgesIncidentTo(vertex);
    int uncolored = 0;
    for (const auto& neighbor : neighbors) {
      if (colors.find(neighbor) == colors.end()) {
        uncolored++;
      }
    }

    VertexInfo info{vertex, static_cast<int>(neighborColors[vertex].size()),
                    uncolored};
    vertexSaturations[vertex] = info;
    queue.push(info);
  }

  void updateNeighborSaturation(const UndirectedGraph<V>& graph,
                                const V& vertex, int color) {
    auto neighbors = graph.edgesIncidentTo(vertex);
    for (const auto& neighbor : neighbors) {
      if (colors.find(neighbor) == colors.end()) {
        neighborColors[neighbor].insert(color);
        updateSaturationDegree(graph, neighbor);
      }
    }
  }

  // The assigned color to each vertex.
  std::unordered_map<V, int> colors;

  // The set of colors assigned to the neighbors of each vertex,
  // to avoid looping over them when determining the next color
  // to use for the current vertex.
  std::unordered_map<V, std::unordered_set<int>> neighborColors;

  // A mapping of vertex to its saturation data.
  std::unordered_map<V, VertexInfo> vertexSaturations;

  // A priority queue to find the next vertex to color.
  std::priority_queue<VertexInfo> queue;
};

}  // namespace graph
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_GRAPH_GRAPH_H_
