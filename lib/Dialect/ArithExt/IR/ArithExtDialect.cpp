#include "lib/Dialect/ArithExt/IR/ArithExtDialect.h"

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define ArithExtOps
#include "lib/Dialect/ArithExt/IR/ArithExtOps.h"

// Generated definitions
#include "lib/Dialect/ArithExt/IR/ArithExtDialect.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/ArithExt/IR/ArithExtOps.cpp.inc"

namespace mlir {
namespace heir {
namespace arith_ext {

void ArithExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/ArithExt/IR/ArithExtOps.cpp.inc"
      >();
}

}  // namespace arith_ext
}  // namespace heir
}  // namespace mlir
