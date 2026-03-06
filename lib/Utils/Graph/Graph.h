#ifndef LIB_UTILS_GRAPH_GRAPH_H_
#define LIB_UTILS_GRAPH_GRAPH_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SetVector.h"          // from @llvm-project
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

  bool hasEdge(V source, V target) {
    if (vertices.count(source) == 0 || vertices.count(target) == 0)
      return false;
    return outEdges[source].find(target) != outEdges[source].end();
  }

  // Returns true iff the given vertex has previously been added to the graph
  // using `AddVertex`.
  bool contains(V vertex) const { return vertices.count(vertex) > 0; }

  bool empty() const { return vertices.empty(); }

  const std::set<V>& getVertices() { return vertices; }
  std::set<std::pair<V, V>> getEdges() const {
    std::set<std::pair<V, V>> result;
    for (const auto& [source, targets] : outEdges) {
      for (const auto& target : targets) {
        result.insert({source, target});
      }
    }
    return result;
  }

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

  // Contract the edge from source to target, thus merging target into source
  // and removing target from the graph. Returns false if the edge doesn't
  // exist. The optional mergeFn is a functor that takes the source and target
  // vertex and is expected to perform custom logic on the vertices that need to
  // be updated as a result of the merge, and is called before the graph is
  // modified. It is called exactly once for the contracted edge if provided.
  //
  // V = struct { string name; }
  // mergeFn = [](V& source, const V& target) { source.name += target.name; }
  //
  // and we contract edge (v1, v2) where v1.name = "foo" and v2.name = "bar",
  // then the mergeFn could update v1.name to "foobar" to preserve some
  // information from both vertices before v2 is removed from the graph.
  bool contractEdge(V source, V target,
                    std::function<void(V&, const V&)> mergeFn = nullptr) {
    if (!hasEdge(source, target)) return false;

    // Merge target into source
    if (mergeFn) mergeFn(source, target);

    // Redirect all outgoing edges from target to source
    for (V succ : edgesOutOf(target)) {
      // Avoid creating a edge to self
      if (succ != source) {
        if (weights.contains({target, succ})) {
          weights[{source, succ}] = weights.at({target, succ});
          weights.erase({target, succ});
        }
        addEdge(source, succ);
      }
    }

    // Redirect all incoming edges to target to source
    for (V pred : edgesInto(target)) {
      // Avoid creating a edge to self
      if (pred != source) {
        if (weights.contains({pred, target})) {
          weights[{pred, source}] = weights.at({pred, target});
          weights.erase({pred, target});
        }
        addEdge(pred, source);
      }
    }
    // Remove the original source→target edge
    outEdges[source].erase(target);
    weights.erase({source, target});

    // Remove stale target references from successors' inEdges
    for (V succ : edgesOutOf(target)) inEdges[succ].erase(target);

    // Remove stale target references from predecessors' outEdges
    for (V pred : edgesInto(target)) outEdges[pred].erase(target);

    vertices.erase(target);
    outEdges.erase(target);
    inEdges.erase(target);

    return true;
  }

  // Returns all strongly connected components using Tarjan's algorithm.
  // Each SCC is returned as a std::set<V> (consistent with getVertices()).
  // Reference implementation:
  // https://cp-algorithms.com/graph/strongly-connected-components.html#implementation_1
  //
  // Returns a vector of set of vertices, where each set represents a strong
  // connected component. The order of the components and the order of vertices
  // within each component are not preserved.
  //
  // TODO(#2736): This implementation does not apply Nuutila's modification.
  // Performance can be bad for larger SCCs. Consider implementing Nuutila's
  // modification if this becomes a bottleneck.
  // https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.strongly_connected_components.html
  std::vector<std::set<V>> getStronglyConnectedComponents() {
    std::vector<std::set<V>> result;
    std::map<V, int> index, lowlink;
    llvm::SetVector<V> stack;
    int currentIndex = 0;

    std::function<void(V)> dfs = [&](V v) {
      index[v] = lowlink[v] = currentIndex++;
      stack.insert(v);

      for (const V& w : edgesOutOf(v)) {
        if (!index.contains(w)) {
          dfs(w);
          lowlink[v] = std::min(lowlink[v], lowlink[w]);
        } else if (stack.contains(w)) {
          lowlink[v] = std::min(lowlink[v], index[w]);
        }
      }

      if (lowlink[v] == index[v]) {
        std::set<V> scc;
        V w;
        do {
          w = stack.back();
          stack.pop_back();
          scc.insert(w);
        } while (w != v);
        result.push_back(std::move(scc));
      }
    };

    for (const V& v : vertices)
      if (!index.contains(v)) dfs(v);

    return result;
  }

  // Condense each strongly connected component into a single vertex, thus
  // making the graph acyclic. The optional mergeFn is a functor that takes the
  // source and target vertex and is expected to perform custom logic on the
  // vertices that need to be updated as a result of the condensation,
  // internally it is consumed by contractEdge when contracting edges between
  // vertices in the same SCC.
  //
  // V = struct { string name; }
  // mergeFn = [](V& source, const V& target) { source.name += target.name; }
  //
  // and we contract edge (v1, v2) where v1.name = "foo" and v2.name = "bar",
  // then the mergeFn could update v1.name to "foobar" to preserve some
  // information from both vertices before v2 is removed from the graph.
  //
  // TODO(#2736): Evaluate if it is more efficient to construct a new graph
  // instead of modifying the existing graph in-place by contracting edges. The
  // current implementation is straightforward but can be inefficient for larger
  // SCCs.
  void condenseGraph(std::function<void(V&, const V&)> mergeFn = nullptr) {
    auto sccs = getStronglyConnectedComponents();

    for (auto& scc : sccs) {
      if (scc.size() <= 1) continue;

      // needs multiples passes as contractEdge can create new edges between
      // vertices in the same SCC that need to be contracted.
      std::set<V> alive(scc.begin(), scc.end());
      while (alive.size() > 1) {
        // needs a snapshot as we modify the alive set while iterating through
        // it.
        std::vector<V> aliveSnapShot(alive.begin(), alive.end());
        for (const V& u : aliveSnapShot) {
          if (!alive.count(u)) continue;
          for (V v : edgesOutOf(u)) {
            if (alive.count(v)) {
              contractEdge(u, v, mergeFn);
              alive.erase(v);
            }
          }
        }
      }
    }
  }

  FailureOr<std::vector<V>> getLongestSourceToSinkPath() {
    auto sinks = getSinks();
    auto sources = getSources();

    std::vector<V> result;

    for (auto& src : sources) {
      for (auto& sink : sinks) {
        if (src == sink) continue;  // isolated node is both source and sink
        auto res = getShortestPath(src, sink);
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
  FailureOr<std::vector<V>> findApproximateCriticalPath() {
    auto result = topologicalSort();
    if (failed(result)) {
      return failure();
    }
    auto sinks = getSinks();
    auto sources = getSources();

    if (result->empty() || result->size() < 3) return *result;
    std::vector<V> cp_result;
    // first node has to be a source node
    if (std::find(sources.begin(), sources.end(), result->at(0)) ==
        sources.end()) {
      return failure();
    }
    // last node has to be a sink node
    if (std::find(sinks.begin(), sinks.end(), result->back()) == sinks.end()) {
      return failure();
    }

    for (uint64_t i = 1; i < result->size() - 1; i++) {
      // skip any source or sink nodes in the CP order
      if (std::find(sources.begin(), sources.end(), result->at(i)) !=
          sources.end())
        continue;
      if (std::find(sinks.begin(), sinks.end(), result->at(i)) != sinks.end())
        continue;
      cp_result.emplace_back(result->at(i));
    }
    cp_result.emplace_back(result->back());
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
  std::set<V> getSources() {
    std::set<V> sources;
    llvm::copy_if(
        vertices, std::inserter(sources, sources.begin()),
        [this](const V& v /*vertex*/) { return this->inEdges[v].size() == 0; });
    return sources;
  }

  /* get vertices in graph with 0 outgoing edges */
  std::set<V> getSinks() {
    std::set<V> sinks;
    llvm::copy_if(vertices, std::inserter(sinks, sinks.begin()),
                  [this](const V& v /*vertex*/) {
                    return this->outEdges[v].size() == 0;
                  });
    return sinks;
  }

  /* get number of incoming edges into vertex */
  uint64_t getInDegree(V vertex) {
    return (!inEdges.count(vertex)) ? 0 : inEdges.at(vertex).size();
  }

  /* get number of outgoing edges from vertex */
  uint64_t getOutDegree(V vertex) {
    return (!outEdges.count(vertex)) ? 0 : outEdges.at(vertex).size();
  }

  FailureOr<std::vector<V>> getShortestPath(V start, V end) {
    std::function<uint64_t(const V&, const V&)> weight_fn =
        [this](const V& start, const V& end) -> uint64_t {
      return this->hasEdge(start, end) ? 1
                                       : std::numeric_limits<uint64_t>::max();
    };
    return getShortestPath(start, end, weight_fn);
  }

  // assume uniform weight for edges; alternatively provide weight matrix
  FailureOr<std::vector<V>> getShortestPath(
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
      const V& curr = q.front();
      q.pop();

      if (curr == end) {
        distance[curr] = 0;
        break;
      }

      if (!visited.insert(curr).second) {
        // backedge
        continue;
      }

      // nothing to reduce on this edge.
      if (!outEdges.count(curr)) {
        continue;
      }

      for (auto& next : outEdges.at(curr)) {
        if (distance[next] > distance[curr] + weight_fn(curr, next)) {
          linkback[curr] = next;
          distance[next] = distance[curr] + weight_fn(curr, next);
        } else if (distance[curr] == 0) {
          linkback[curr] = next;
        }
        q.push(next);
      }
    }

    if (distance[end] != std::numeric_limits<uint64_t>::max()) {
      // some path found
      auto curr = start;
      while (curr != end) {
        path.push_back(curr);
        curr = linkback[curr];
      }
      path.push_back(end);
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

  bool setWeight(V start, V end, uint64_t weight) {
    if (!gWeights.count(start)) {
      gWeights[start] = {std::make_pair(end, weight)};
      return true;
    }
    return gWeights[start].insert(std::make_pair(end, weight)).second;
  }

  // get weight
  uint64_t operator()(V start, V end) { return 0; }
};

// An undirected graph data structure.
//
// Parameter `V` is the vertex type, which is expected to be cheap to copy.
template <typename V>
class UndirectedGraph {
 public:
  // Adds a vertex to the graph
  void addVertex(V vertex) { vertices.insert(vertex); }

  bool hasEdge(V source, V target) {
    if (vertices.count(source) == 0 || vertices.count(target) == 0)
      return false;
    return edges[source].find(target) != edges[source].end();
  }

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
