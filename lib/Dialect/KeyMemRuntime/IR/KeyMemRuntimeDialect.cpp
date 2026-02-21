#include "lib/Dialect/KeyMemRuntime/IR/KeyMemRuntimeDialect.h"

#include "lib/Dialect/KeyMemRuntime/IR/KeyMemRuntimeDialect.cpp.inc"
#include "lib/Dialect/KeyMemRuntime/IR/KeyMemRuntimeOps.h"
#include "lib/Dialect/KeyMemRuntime/IR/KeyMemRuntimeTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"     // from @llvm-project
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/KeyMemRuntime/IR/KeyMemRuntimeTypes.cpp.inc"

namespace mlir {
namespace heir {
namespace kmrt {

void KeyMemRuntimeDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/KeyMemRuntime/IR/KeyMemRuntimeTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/KeyMemRuntime/IR/KeyMemRuntimeOps.cpp.inc"
      >();
}

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir
