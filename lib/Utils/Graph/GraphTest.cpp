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
