#include <unordered_map>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Graph/Graph.h"
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace graph {
namespace {

using ::testing::UnorderedElementsAre;

Graph<int> make_levels_graph() {
  Graph<int> graph;
  graph.addVertex(0);
  graph.addVertex(1);
  graph.addVertex(2);
  graph.addVertex(3);
  graph.addVertex(4);
  graph.addVertex(5);
  graph.addVertex(6);
  graph.addVertex(7);
  graph.addVertex(8);
  graph.addVertex(9);
  graph.addVertex(10);
  graph.addEdge(0, 5);
  graph.addEdge(1, 6);
  graph.addEdge(2, 7);
  graph.addEdge(3, 8);
  graph.addEdge(4, 9);
  graph.addEdge(5, 6);
  graph.addEdge(6, 7);
  graph.addEdge(7, 8);
  graph.addEdge(8, 9);
  graph.addEdge(9, 10);
  return graph;
}

TEST(GraphUtilsTest, getEdges) {
  auto graph = make_levels_graph();
  ASSERT_EQ(graph.getEdges().size(), 10);
}

TEST(GraphUtilsTest, SourceAndSink) {
  auto graph = make_levels_graph();
  ASSERT_EQ(graph.getSources().size(), 5);
  ASSERT_EQ(graph.getSinks().size(), 1);
}

TEST(GraphUtilsTest, DegreeInAndOut) {
  auto graph = make_levels_graph();
  ASSERT_EQ(graph.topologicalSort()->size(), 11);
  ASSERT_EQ(graph.getInDegree(0), 0);
  ASSERT_EQ(graph.getInDegree(5), 1);
  ASSERT_EQ(graph.getOutDegree(9), 1);
  ASSERT_EQ(graph.getOutDegree(10), 0);
}

TEST(GraphUtilsTest, LongPathsA) {
  auto graph = make_levels_graph();
  auto resultpath = graph.getShortestPath(1, 10);
  ASSERT_TRUE(succeeded(resultpath));
  ASSERT_EQ((*resultpath).size(), 6);
  ASSERT_EQ((*resultpath)[0], 1);
  ASSERT_EQ((*resultpath)[1], 6);
  ASSERT_EQ((*resultpath)[2], 7);
  ASSERT_EQ((*resultpath)[3], 8);
  ASSERT_EQ((*resultpath)[4], 9);
  ASSERT_EQ((*resultpath)[5], 10);
}

TEST(GraphUtilsTest, LongPathsB) {
  auto graph = make_levels_graph();
  graph.addEdge(1, 9);
  auto resultpath = graph.getShortestPath(1, 10);
  ASSERT_TRUE(succeeded(resultpath));
  ASSERT_EQ((*resultpath).size(), 3);
  ASSERT_EQ((*resultpath)[0], 1);
  ASSERT_EQ((*resultpath)[1], 9);
  ASSERT_EQ((*resultpath)[2], 10);
}

TEST(GraphUtilsTest, LongPathsC) {
  auto graph = make_levels_graph();
  graph.addEdge(1, 9);
  auto resultpath = graph.getShortestPath(1, 0);
  ASSERT_TRUE(failed(resultpath));
}

TEST(GraphUtilsTest, CriticalPath) {
  auto graph = make_levels_graph();
  auto resultpath = graph.findApproximateCriticalPath();
  ASSERT_TRUE(succeeded(resultpath));
  ASSERT_EQ((*resultpath).size(), 6);
  ASSERT_EQ((*resultpath)[0], 5);
  ASSERT_EQ((*resultpath)[1], 6);
  ASSERT_EQ((*resultpath)[2], 7);
  ASSERT_EQ((*resultpath)[3], 8);
  ASSERT_EQ((*resultpath)[4], 9);
  ASSERT_EQ((*resultpath)[5], 10);
}

TEST(ContractEdgeTest, SimpleContraction) {
  // Before: 0 → 1 → 2
  // Contract (0, 1) → merge 1 into 0
  // After:  0 → 2   (0 now has 1's successor)
  Graph<int> graph;
  graph.addVertex(0);
  graph.addVertex(1);
  graph.addVertex(2);
  graph.addEdge(0, 1);
  graph.addEdge(1, 2);

  graph.contractEdge(0, 1);

  EXPECT_FALSE(graph.contains(1));
  EXPECT_TRUE(graph.contains(0));
  EXPECT_TRUE(graph.contains(2));
  EXPECT_TRUE(graph.hasEdge(0, 2));
  EXPECT_FALSE(graph.hasEdge(0, 1));
}

TEST(ContractEdgeTest, PreservesOtherEdges) {
  // Before: 0 → 1 → 3
  //           ↘   ↗
  //             2
  // Contract (1, 3) → merge 3 into 1
  // After:  0 → 1,  0 → 2 → 1
  Graph<int> graph;
  graph.addVertex(0);
  graph.addVertex(1);
  graph.addVertex(2);
  graph.addVertex(3);
  graph.addEdge(0, 1);
  graph.addEdge(0, 2);
  graph.addEdge(1, 3);
  graph.addEdge(2, 3);

  graph.contractEdge(1, 3);

  EXPECT_FALSE(graph.contains(3));
  EXPECT_TRUE(graph.hasEdge(0, 1));
  EXPECT_TRUE(graph.hasEdge(0, 2));
  EXPECT_TRUE(graph.hasEdge(2, 1));
  EXPECT_FALSE(graph.hasEdge(1, 3));
  EXPECT_EQ(graph.getVertices().size(), 3);
}

TEST(ContractEdgeTest, NoSelfLoop) {
  // Before: 0 → 1 → 2
  //          ↘ → → ↗
  // Contract (0, 1): 1's successor is 2, which 0 already points to.
  // After:  0 → 2   (no self-loop, no duplicate edge)
  Graph<int> graph;
  graph.addVertex(0);
  graph.addVertex(1);
  graph.addVertex(2);
  graph.addEdge(0, 1);
  graph.addEdge(1, 2);
  graph.addEdge(0, 2);

  graph.contractEdge(0, 1);

  EXPECT_FALSE(graph.contains(1));
  EXPECT_TRUE(graph.hasEdge(0, 2));
  EXPECT_FALSE(graph.hasEdge(0, 0));    // no self-loop
  EXPECT_EQ(graph.getOutDegree(0), 1);  // only one edge to 2
}

// Functor that records which pairs were merged and keeps the source vertex.
struct MergeRecorder {
  std::vector<std::pair<int, int>> calls;
  void operator()(int& source, const int& target) {
    calls.push_back({source, target});
  }
};

TEST(ContractEdgeTest, LargerGraphWithFunctor) {
  // Graph:
  //
  //  0 → 1 → 2 → 3 → 7
  //      ↓       ↑
  //      4 → 5 → 6
  //
  // Contract edge (2, 3):
  //   - 3's successor  (3→7) redirected  → 2→7
  //   - 3's predecessor (6→3) redirected → 6→2
  //   - 3 removed
  //
  // After:
  //  0 → 1 → → → 2 → 7
  //      ↓       ↑
  //      4 → 5 → 6

  Graph<int> graph;
  for (int i = 0; i <= 7; i++) graph.addVertex(i);
  graph.addEdge(0, 1);
  graph.addEdge(1, 2);
  graph.addEdge(1, 4);
  graph.addEdge(2, 3);
  graph.addEdge(3, 7);
  graph.addEdge(4, 5);
  graph.addEdge(5, 6);
  graph.addEdge(6, 3);

  MergeRecorder recorder;
  bool result = graph.contractEdge(2, 3, std::ref(recorder));

  EXPECT_TRUE(result);

  // Functor called exactly once with (source=2, target=3)
  ASSERT_EQ(recorder.calls.size(), 1);
  EXPECT_EQ(recorder.calls[0], std::make_pair(2, 3));

  // 3 is gone
  EXPECT_FALSE(graph.contains(3));
  EXPECT_EQ(graph.getVertices().size(), 7);

  // 3's outgoing edge (3→7) redirected to 2→7
  EXPECT_TRUE(graph.hasEdge(2, 7));

  // 3's incoming edge (6→3) redirected to 6→2
  EXPECT_TRUE(graph.hasEdge(6, 2));

  // Original edges preserved
  EXPECT_TRUE(graph.hasEdge(0, 1));
  EXPECT_TRUE(graph.hasEdge(1, 2));
  EXPECT_TRUE(graph.hasEdge(1, 4));
  EXPECT_TRUE(graph.hasEdge(4, 5));
  EXPECT_TRUE(graph.hasEdge(5, 6));

  // Contracted edge and all edges to 3 gone
  EXPECT_FALSE(graph.hasEdge(2, 3));
  EXPECT_FALSE(graph.hasEdge(6, 3));
  EXPECT_FALSE(graph.hasEdge(3, 7));

  // No self-loop on 2
  EXPECT_FALSE(graph.hasEdge(2, 2));
}

TEST(ContractEdgeTest, ReturnsFalseIfEdgeAbsent) {
  Graph<int> graph;
  graph.addVertex(0);
  graph.addVertex(1);
  graph.addVertex(2);
  graph.addEdge(0, 1);

  MergeRecorder recorder;
  // Edge (1, 0) doesn't exist (wrong direction)
  EXPECT_FALSE(graph.contractEdge(1, 0, std::ref(recorder)));
  // Edge (0, 2) doesn't exist
  EXPECT_FALSE(graph.contractEdge(0, 2, std::ref(recorder)));

  // Nothing was merged, graph unchanged
  EXPECT_EQ(recorder.calls.size(), 0);
  EXPECT_EQ(graph.getVertices().size(), 3);
  EXPECT_TRUE(graph.hasEdge(0, 1));
}

// getStronglyConnectedComponents tests
TEST(SCCTest, DAGEachNodeIsOwnSCC) {
  // 0 → 1 → 2 → 3  (no cycles)
  // Every node is its own SCC.
  Graph<int> g;
  for (int i = 0; i < 4; i++) g.addVertex(i);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 3);

  auto sccs = g.getStronglyConnectedComponents();
  EXPECT_EQ(sccs.size(), 4);
  EXPECT_THAT(sccs, UnorderedElementsAre(std::set<int>{0}, std::set<int>{1},
                                         std::set<int>{2}, std::set<int>{3}));
}

TEST(SCCTest, SingleCycleIsOneSCC) {
  // 0 → 1 → 2 → 0  (one big cycle)
  Graph<int> g;
  for (int i = 0; i < 3; i++) g.addVertex(i);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 0);

  auto sccs = g.getStronglyConnectedComponents();
  ASSERT_EQ(sccs.size(), 1);
  EXPECT_EQ(sccs[0], (std::set<int>{0, 1, 2}));
}

TEST(SCCTest, SingleVertex) {
  Graph<int> g;
  g.addVertex(42);

  auto sccs = g.getStronglyConnectedComponents();
  ASSERT_EQ(sccs.size(), 1);
  EXPECT_EQ(sccs[0], (std::set<int>{42}));
}

TEST(SCCTest, TwoCyclesConnectedByBridge) {
  // Example graph:
  //
  // 0 → 1 → 2 → 3 → 4
  // ↑       ↓     ↖ ↓
  // └────── 0       5
  //
  Graph<int> g;
  for (int i = 0; i < 6; i++) g.addVertex(i);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 0);
  g.addEdge(2, 3);
  g.addEdge(3, 4);
  g.addEdge(4, 5);
  g.addEdge(5, 3);

  auto sccs = g.getStronglyConnectedComponents();
  ASSERT_EQ(sccs.size(), 2);
  EXPECT_THAT(sccs, UnorderedElementsAre(std::set<int>{0, 1, 2},
                                         std::set<int>{3, 4, 5}));
}

TEST(SCCTest, MixedSCCSizes) {
  // Example graph:
  //
  // 0 → 1 → 2 → 3 → 4
  // ↑   │       ↑   │
  // └───┘       └───┘
  //
  Graph<int> g;
  for (int i = 0; i < 5; i++) g.addVertex(i);
  g.addEdge(0, 1);
  g.addEdge(1, 0);
  g.addEdge(1, 2);
  g.addEdge(2, 3);
  g.addEdge(3, 4);
  g.addEdge(4, 3);

  auto sccs = g.getStronglyConnectedComponents();
  ASSERT_EQ(sccs.size(), 3);
  EXPECT_THAT(sccs, UnorderedElementsAre(std::set<int>{0, 1}, std::set<int>{2},
                                         std::set<int>{3, 4}));
}

TEST(SCCTest, DisconnectedGraph) {
  // Three isolated vertices — each its own SCC.
  Graph<int> g;
  g.addVertex(10);
  g.addVertex(20);
  g.addVertex(30);

  auto sccs = g.getStronglyConnectedComponents();
  ASSERT_EQ(sccs.size(), 3);
  EXPECT_THAT(sccs, UnorderedElementsAre(std::set<int>{10}, std::set<int>{20},
                                         std::set<int>{30}));
}

// condenseGraph tests
TEST(CondenseTest, DAGCondenseIsIdentity) {
  Graph<int> g;
  for (int i = 0; i < 4; i++) g.addVertex(i);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 3);

  g.condenseGraph();

  EXPECT_EQ(g.getVertices().size(), 4);
  EXPECT_TRUE(g.hasEdge(0, 1));
  EXPECT_TRUE(g.hasEdge(1, 2));
  EXPECT_TRUE(g.hasEdge(2, 3));
  EXPECT_FALSE(g.hasEdge(0, 2));
}

TEST(CondenseTest, CycleCollapsedToOneNode) {
  // 3
  // ↑
  // 0 → 1 → 2
  //  ↖ ___ ↙

  Graph<int> g;
  g.addVertex(0);
  g.addVertex(1);
  g.addVertex(2);
  g.addVertex(10);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 0);
  g.addEdge(0, 10);

  MergeRecorder recorder;

  // Map each SCC to the sum of its members (unique within this graph).
  g.condenseGraph(std::ref(recorder));

  ASSERT_EQ(recorder.calls.size(), 2);
  ASSERT_EQ(g.getVertices().size(), 2);
  EXPECT_TRUE(g.hasEdge(2, 10));
  EXPECT_FALSE(g.hasEdge(2, 2));
  EXPECT_THAT(recorder.calls,
              UnorderedElementsAre(std::make_pair(0, 1), std::make_pair(2, 0)));
}

TEST(CondenseTest, NoSelfLoopsAfterCondense) {
  // After condensing, the result must be a DAG (no cycles, no self-loops).
  Graph<int> g;
  for (int i = 0; i < 6; i++) g.addVertex(i);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 0);
  g.addEdge(2, 3);
  g.addEdge(3, 4);
  g.addEdge(4, 5);
  g.addEdge(5, 3);

  g.condenseGraph();

  // Result must be acyclic.
  EXPECT_TRUE(succeeded(g.topologicalSort()));

  ASSERT_EQ(g.getVertices().size(), 2);
  EXPECT_FALSE(g.hasEdge(2, 2));
  EXPECT_FALSE(g.hasEdge(5, 5));
  EXPECT_TRUE(g.hasEdge(2, 5));
}

TEST(LevelSortTest, SimpleGraphLevelSort) {
  // Example graph:
  //       ↗ 2 ↘
  // 0 → 1 → 3 → 4
  //   ↘ → → → ↗
  //
  // Level divisions:
  // 0 | 1 | 2 | 3
  //
  // Level 0: node 0
  // Level 1: node 1
  // Level 2: node 2, 3
  // Level 3: node 4

  Graph<int> graph;
  graph.addVertex(0);
  graph.addVertex(1);
  graph.addVertex(2);
  graph.addVertex(3);
  graph.addVertex(4);
  EXPECT_TRUE(graph.addEdge(0, 1));
  EXPECT_TRUE(graph.addEdge(1, 2));
  EXPECT_TRUE(graph.addEdge(1, 3));
  EXPECT_TRUE(graph.addEdge(1, 4));
  EXPECT_TRUE(graph.addEdge(2, 4));
  EXPECT_TRUE(graph.addEdge(3, 4));

  auto levelSorted = graph.sortGraphByLevels();
  EXPECT_TRUE(succeeded(levelSorted));
  std::vector<std::vector<int>> levelUnwrapped = levelSorted.value();
  EXPECT_EQ(levelUnwrapped.size(), 4);
  EXPECT_THAT(levelUnwrapped[0], UnorderedElementsAre(0));
  EXPECT_THAT(levelUnwrapped[1], UnorderedElementsAre(1));
  EXPECT_THAT(levelUnwrapped[2], UnorderedElementsAre(2, 3));
  EXPECT_THAT(levelUnwrapped[3], UnorderedElementsAre(4));
}

TEST(LevelSortTest, MultiInputGraphLevelSort) {
  // Example graph with multiple inputs:
  // 0 → 5 → 6 → 7 → 8 → 9 → 10
  //     1 ↗    ↑    ↑   ↑
  //         2 ↗     ↑   ↑
  //             3 ↗     ↑
  //                 4 ↗
  //
  // Level divisions:
  // 0 | 1 | 2 | 3 | 4 | 5 | 6
  // Level 0: node 0
  // Level 1: node 1, 5
  // Level 2: node 2, 6
  // Level 3: node 3, 7
  // Level 4: node 4, 8
  // Level 5: node 9
  // Level 6: node 10

  auto graph = make_levels_graph();
  auto levelSorted = graph.sortGraphByLevels();
  EXPECT_TRUE(succeeded(levelSorted));
  std::vector<std::vector<int>> levelUnwrapped = levelSorted.value();
  EXPECT_EQ(levelUnwrapped.size(), 7);
  EXPECT_THAT(levelUnwrapped[0], UnorderedElementsAre(0));
  EXPECT_THAT(levelUnwrapped[1], UnorderedElementsAre(5, 1));
  EXPECT_THAT(levelUnwrapped[2], UnorderedElementsAre(2, 6));
  EXPECT_THAT(levelUnwrapped[3], UnorderedElementsAre(3, 7));
  EXPECT_THAT(levelUnwrapped[4], UnorderedElementsAre(4, 8));
  EXPECT_THAT(levelUnwrapped[5], UnorderedElementsAre(9));
  EXPECT_THAT(levelUnwrapped[6], UnorderedElementsAre(10));
}

TEST(LevelSortTest, MultiOutputGraphLevelSort) {
  // Example graph with multiple outputs:
  // 0 → 1 → 2 → 3 → 4 → 5
  //       ↘   ↘   ↘  ↘  6
  //         ↘   ↘   ↘ → 7
  //           ↘   ↘ → → 8
  //             ↘ → → → 9
  //
  // Level divisions:
  // 0 | 1 | 2 | 3 | 4 | 5
  // Level 0: node 0
  // Level 1: node 1
  // Level 2: node 2
  // Level 3: node 3
  // Level 4: node 4
  // Level 5: node 5, 6, 7, 8, 9

  Graph<int> graph;
  graph.addVertex(0);
  graph.addVertex(1);
  graph.addVertex(2);
  graph.addVertex(3);
  graph.addVertex(4);
  graph.addVertex(5);
  graph.addVertex(6);
  graph.addVertex(7);
  graph.addVertex(8);
  graph.addVertex(9);
  graph.addEdge(0, 1);
  graph.addEdge(1, 2);
  graph.addEdge(2, 3);
  graph.addEdge(3, 4);
  graph.addEdge(4, 5);
  graph.addEdge(4, 6);
  graph.addEdge(1, 9);
  graph.addEdge(2, 8);
  graph.addEdge(3, 7);

  auto levelSorted = graph.sortGraphByLevels();
  EXPECT_TRUE(succeeded(levelSorted));
  std::vector<std::vector<int>> levelUnwrapped = levelSorted.value();
  EXPECT_EQ(levelUnwrapped.size(), 6);
  EXPECT_THAT(levelUnwrapped[0], UnorderedElementsAre(0));
  EXPECT_THAT(levelUnwrapped[1], UnorderedElementsAre(1));
  EXPECT_THAT(levelUnwrapped[2], UnorderedElementsAre(2));
  EXPECT_THAT(levelUnwrapped[3], UnorderedElementsAre(3));
  EXPECT_THAT(levelUnwrapped[4], UnorderedElementsAre(4));
  EXPECT_THAT(levelUnwrapped[5], UnorderedElementsAre(5, 6, 7, 8, 9));
}

TEST(GraphColorTest, SimpleGraph) {
  // Example graph:
  //       / 2 \
  // 0 - 1 - 3 - 4
  //   \ - - - /
  UndirectedGraph<int> graph;
  graph.addVertex(0);
  graph.addVertex(1);
  graph.addVertex(2);
  graph.addVertex(3);
  graph.addVertex(4);
  EXPECT_TRUE(graph.addEdge(0, 1));
  EXPECT_TRUE(graph.addEdge(1, 2));
  EXPECT_TRUE(graph.addEdge(1, 3));
  EXPECT_TRUE(graph.addEdge(2, 4));
  EXPECT_TRUE(graph.addEdge(3, 4));

  GreedyGraphColoring<int> greedy;
  std::unordered_map<int, int> colors = greedy.color(graph);
  // assertions in visitation order
  EXPECT_EQ(colors[1], 0);
  EXPECT_EQ(colors[4], 0);
  EXPECT_EQ(colors[2], 1);
  EXPECT_EQ(colors[3], 1);
  EXPECT_EQ(colors[0], 1);
}

TEST(GraphColorTest, CompleteGraph) {
  UndirectedGraph<int> graph;
  graph.addVertex(0);
  graph.addVertex(1);
  graph.addVertex(2);
  graph.addVertex(3);
  graph.addVertex(4);
  EXPECT_TRUE(graph.addEdge(0, 1));
  EXPECT_TRUE(graph.addEdge(0, 2));
  EXPECT_TRUE(graph.addEdge(0, 3));
  EXPECT_TRUE(graph.addEdge(0, 4));
  EXPECT_TRUE(graph.addEdge(1, 2));
  EXPECT_TRUE(graph.addEdge(1, 3));
  EXPECT_TRUE(graph.addEdge(1, 4));
  EXPECT_TRUE(graph.addEdge(2, 3));
  EXPECT_TRUE(graph.addEdge(2, 4));
  EXPECT_TRUE(graph.addEdge(3, 4));

  GreedyGraphColoring<int> greedy;
  std::unordered_map<int, int> colors = greedy.color(graph);
  EXPECT_EQ(colors[0], 0);
  EXPECT_EQ(colors[1], 1);
  EXPECT_EQ(colors[2], 2);
  EXPECT_EQ(colors[3], 3);
  EXPECT_EQ(colors[4], 4);
}

TEST(DSATURColorTest, StarGraph) {
  // Center vertex connected to 4 leaves
  UndirectedGraph<int> graph;
  for (int i = 0; i < 5; i++) graph.addVertex(i);
  for (int i = 1; i < 5; i++) EXPECT_TRUE(graph.addEdge(0, i));

  GreedyGraphColoring<int> greedy;
  auto colors = greedy.color(graph);
  EXPECT_EQ(colors[0], 0);  // Center colored first
  for (int i = 1; i < 5; i++) {
    EXPECT_EQ(colors[i], 1);  // All leaves same color
    EXPECT_NE(colors[0], colors[i]);
  }
}

}  // namespace
}  // namespace graph
}  // namespace heir
}  // namespace mlir
