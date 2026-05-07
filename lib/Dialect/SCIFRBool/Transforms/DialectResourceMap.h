#ifndef LIB_DIALECT_SCIFRBOOL_TRANSFORMS_DIALECTRESOURCEMAP_H_
#define LIB_DIALECT_SCIFRBOOL_TRANSFORMS_DIALECTRESOURCEMAP_H_

#include <cstdint>
#include <map>
#include <string>

#include "lib/Utils/Graph/Graph.h"
#include "mlir/include/mlir/IR/Dialect.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"  // from @llvm-project

namespace mlir {
namespace cornami {
using OpCoreCountAndLatencyT = std::pair<uint64_t, uint64_t>;
using DialectResourceMap = std::map<std::string, uint64_t>;
using DialectStringBootstrapAndKeySwitchMap =
    std::map<std::string, std::array<bool, 2>>;  // does operator need bootstrap
                                                 // and keyswitch ?
// Operations here with initial Counts
void dump(heir::graph::Graph<Operation*>& graph);
}  // namespace cornami
}  // namespace mlir

#endif  // LIB_DIALECT_SCIFRBOOL_TRANSFORMS_DIALECTRESOURCEMAP_H_
