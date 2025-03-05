#include "lib/Dialect/LWE/IR/LWEDialect.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"

// IWYU pragma: begin_keep
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
// IWYU pragma: end_keep

// Generated definitions
#include "lib/Dialect/LWE/IR/LWEDialect.cpp.inc"
#include "lib/Dialect/LWE/IR/LWEEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/LWE/IR/LWEAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/LWE/IR/LWETypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/LWE/IR/LWEOps.cpp.inc"

namespace mlir {
namespace heir {
namespace lwe {

void LWEDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/LWE/IR/LWEAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/LWE/IR/LWETypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/LWE/IR/LWEOps.cpp.inc"
      >();
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
