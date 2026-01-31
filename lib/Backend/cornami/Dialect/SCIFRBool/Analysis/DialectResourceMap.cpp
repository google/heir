#include "lib/Backend/cornami/Dialect/SCIFRBool/Analysis/DialectResourceMap.h"

namespace mlir {
namespace cornami {

cornami::DialectResourceMap make_HELayers_resource_map() {
  return cornami::DialectResourceMap{{"ADD", 0},
                                     {"SUB", 0},
                                     {"MUL", 0},
                                     {"NEG", 0},
                                     {"MUL_RAW", 0},
                                     {"META_SHAPE", 0},
                                     {"ROTATE", 0},
                                     {"PMUL", 0},
                                     {"PSUB", 0},
                                     {"RELIN", 0},
                                     {"READ", 0},
                                     {"RESC", 0},
                                     {"LABEL", 0},
                                     {"BOOTSTRAP", 0},
                                     {"BOOTSTRAPREAL", 0},
                                     {"SQ", 0},
                                     {"SET", 0},
                                     {"SET_CI", 0},
                                     {"SELECTOR", 0},
                                     {"ADD_SCALAR_DOUBLE", 0},
                                     {"ADD_SCALAR_INT", 0},
                                     {"MUL_SCALAR_DOUBLE", 0},
                                     {"OUTPUT", 0},
                                     {"PARAM", 0},
                                     {"None", 0}};
}

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
