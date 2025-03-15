#include "lib/Dialect/BGV/IR/BGVDialect.h"

// IWYU pragma: begin_keep
#include <optional>

#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVEnums.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
// IWYU pragma: end_keep

// Generated definitions
#include "lib/Dialect/BGV/IR/BGVDialect.cpp.inc"
#include "lib/Dialect/BGV/IR/BGVEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/BGV/IR/BGVAttributes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/BGV/IR/BGVOps.cpp.inc"

namespace mlir {
namespace heir {
namespace bgv {

//===----------------------------------------------------------------------===//
// BGV dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void BGVDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/BGV/IR/BGVAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/BGV/IR/BGVOps.cpp.inc"
      >();
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
