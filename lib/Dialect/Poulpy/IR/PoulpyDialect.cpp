#include "lib/Dialect/Poulpy/IR/PoulpyDialect.h"

#include "lib/Dialect/Poulpy/IR/PoulpyOps.h"
#include "lib/Dialect/Poulpy/IR/PoulpyTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// Generated definitions
#include "lib/Dialect/Poulpy/IR/PoulpyDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Poulpy/IR/PoulpyTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/Poulpy/IR/PoulpyOps.cpp.inc"

namespace mlir {
namespace heir {
namespace poulpy {

void PoulpyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Poulpy/IR/PoulpyTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Poulpy/IR/PoulpyOps.cpp.inc"
      >();
}

}  // namespace poulpy
}  // namespace heir
}  // namespace mlir
