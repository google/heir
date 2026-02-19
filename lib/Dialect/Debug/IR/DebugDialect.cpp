#include "lib/Dialect/Debug/IR/DebugDialect.h"

#include "lib/Dialect/Debug/IR/DebugOps.h"

// Generated definitions
#include "lib/Dialect/Debug/IR/DebugDialect.cpp.inc"

namespace mlir {
namespace heir {
namespace debug {

void DebugDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Debug/IR/DebugOps.cpp.inc"
      >();
}

}  // namespace debug
}  // namespace heir
}  // namespace mlir
