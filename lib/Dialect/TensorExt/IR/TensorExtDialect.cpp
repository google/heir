#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"

#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define TensorExtOps
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"

// Generated definitions
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/TensorExt/IR/TensorExtOps.cpp.inc"

namespace mlir {
namespace heir {
namespace tensor_ext {

void TensorExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/TensorExt/IR/TensorExtOps.cpp.inc"
      >();
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
