#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"

#include <optional>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"

// IWYU pragma: begin_keep
#include "llvm/include/llvm/ADT/TypeSwitch.h"    // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
// IWYU pragma: end_keep

#include "lib/Dialect/Openfhe/IR/OpenfheDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/Openfhe/IR/OpenfheOps.cpp.inc"

namespace mlir {
namespace heir {
namespace openfhe {

void OpenfheDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Openfhe/IR/OpenfheOps.cpp.inc"
      >();
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
