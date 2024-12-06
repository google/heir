#ifndef LIB_UTILS_GRAPH_GRAPH_H_
#define LIB_UTILS_GRAPH_GRAPH_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
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

  bool addEdge(V source, V target, int weight) {
    if (addEdge(source, target)) {
      weights[{source, target}] = weight;
      return true;
    }
    return false;
  }

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

  FailureOr<std::vector<V>> get_longest_source_to_sink_path() {
    auto sinks = get_sinks();
    auto sources = get_sources();

    std::function<uint64_t(V, V)> weight_fn = [](V start, V end) -> uint64_t {
      return 1;
    };

    std::vector<V> result;

    for (auto& src : sources) {
      for (auto& sink : sinks) {
        auto res = get_shortest_path(src, sink);
        if (succeeded(res)) {
          if (result.size() < res.size()) {
            result = res;
          }
        }
      }
    }
    return result;
  }

  // calculate the longest path in topologically sorted but restrict to first
  // source (remove rest) and last sink (remove rest) - this is approximate
  // critical path
  FailureOr<std::vector<V>> approximate_critical_path() {
    auto result = topologicalSort();
    if (failed(result)) {
      return failure();
    }
    auto sinks = get_sinks();
    auto sources = get_sources();

    if (result.empty() || result.size() < 3) return result;
    std::vector<V> cp_result;
    // first node has to be a source node
    if (std::find(sources.begin(), sources.end(), result.at(0)) ==
        sources.end()) {
      return failure();
    }
    // last node has to be a sink node
    if (std::find(sinks.begin(), sinks.end(), result.back()) == sinks.end()) {
      return failure();
    }

    cp_result.emplace_back(result.at(0));
    for (uint64_t i = 1; i < result.size() - 1; i++) {
      // skip any source or sink nodes in the CP order
      if (std::find(sources.begin(), sources.end(), result.at(i)) !=
          sources.end())
        continue;
      if (std::find(sinks.begin(), sinks.end(), result.at(i)) != sinks.end())
        continue;
      cp_result.emplace_back(result.at(i));
    }
    cp_result.emplace_back(result.back());
    return cp_result;
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

  /* get vertices in graph with 0 incoming edges */
  std::set<V> get_sources() {
    std::set<V> sources;
    for (V& v : vertices) {
      if (get_in_edges(v) == 0) {
        sources.push_back(v);
      }
    }
    return sources;
  }

  /* get vertices in graph with 0 outgoing edges */
  std::set<V> get_sinks() {
    std::set<V> sinks;
    for (V& v : vertices) {
      if (get_out_edges(v) == 0) {
        sinks.push_back(v);
      }
    }
    return sinks;
  }

  /* get number of incoming edges into vertex */
  uint64_t get_in_degree(V vertex) {
    return inEdges.count(vertex) ? 0 : inEdges.at(vertex).size();
  }

  /* get number of outgoing edges from vertex */
  uint64_t get_out_degree(V vertex) {
    return outEdges.count(vertex) ? 0 : outEdges.at(vertex).size();
  }

  // assume uniform weight for edges; alternatively provide weight matrix
  FailureOr<std::vector<V>> get_shortest_path(
      V start, V end, std::function<uint64_t(V, V)> weight_fn) {
    std::vector<V> path;
    std::queue<V> q;

    std::set<V> visited;
    std::map<V, uint64_t> distance;  // from @start node
    std::map<V, V> linkback;
    for (auto& v : vertices) {
      distance[v] = std::numeric_limits<uint64_t>::max();
    }
    distance[start] = 0;

    q.push(start);

    while (!q.empty()) {
      V curr = q.front();
      q.pop();

      if (!visited.insert(curr).second) {
        // backend found
        continue;
      }

      // nothing to reduce on this edge.
      if (!outEdges.count(curr)) continue;

      for (auto& next : outEdges.at(curr)) {
        if (distance[curr] > distance[next] + weight_fn(curr, next)) {
          linkback[curr] = next;
          distance[curr] = distance[next] + weight_fn(curr, next);
        }
        q.push(next);
      }
    }

    if (distance[end] != std::numeric_limits<uint64_t>::max()) {
      // some path found
      auto curr = end;
      while (linkback[curr] != start) {
        path.insert(path.begin(), curr);
        curr = linkback[curr];
      }
    } else {
      return failure();
    }
    return path;
  }

 private:
  std::set<V> vertices;
  std::map<V, std::set<V>> outEdges;
  std::map<V, std::set<V>> inEdges;
  std::map<std::pair<V, V>, int> weights;
};

template <typename V>
class GraphWeight {
  std::map<V, std::set<std::pair<V, uint64_t>>> gWeights;

  bool set_weight(V start, V end, uint64_t weight) {
    if (!gWeights.count(start)) {
      gWeights[start] = {std::make_pair(end, weight)};
      return true;
    }
    return gWeights[start].insert(std::make_pair(end, weight)).second;
  }

  // get weight
  uint64_t operator()(V start, V end) { return 0; }
};

}  // namespace graph
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_GRAPH_GRAPH_H_
