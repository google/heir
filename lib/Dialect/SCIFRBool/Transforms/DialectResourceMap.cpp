#include "lib/Dialect/SCIFRBool/Transforms/DialectResourceMap.h"

namespace mlir {
namespace cornami {

void dump(heir::graph::Graph<Operation*>& graph) {
  size_t n_vertices = graph.getVertices().size();
  size_t n_out_edges = 0;
  size_t n_in_edges = 0;
  for (auto vertexptr : graph.getVertices()) {
    n_out_edges += graph.getOutDegree(vertexptr);
  }
  for (auto vertexptr : graph.getVertices()) {
    n_in_edges += graph.getInDegree(vertexptr);
  }
  auto n_sources = graph.getSources().size();
  auto n_sinks = graph.getSinks().size();
  llvm::outs() << "====== graph @ display =====\n";
  llvm::outs() << "vertices = " << n_vertices << ", in-edges = " << n_in_edges
               << ", out-edges = " << n_out_edges << ", sinks = " << n_sinks
               << ", source = " << n_sources << "\n";
  llvm::outs() << "====== xxxxxxxxxxxxxxx =====\n";
  return;
}

}  // namespace cornami
}  // namespace mlir
