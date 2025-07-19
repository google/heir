#include "lib/Dialect/MathExt/IR/MathExtDialect.h"

#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define MathExtOps

#include "lib/Dialect/MathExt/IR/MathExtOps.h"

// Generated definitions
#include "lib/Dialect/MathExt/IR/MathExtDialect.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/MathExt/IR/MathExtOps.cpp.inc"

namespace mlir {
namespace heir {
namespace math_ext {

void MathExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/MathExt/IR/MathExtOps.cpp.inc"
      >();
}

}  // namespace math_ext
}  // namespace heir
}  // namespace mlir
