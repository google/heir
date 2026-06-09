#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"

// IWYU pragma: begin_keep
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
// IWYU pragma: end_keep

// Generated definitions
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.cpp.inc"

namespace mlir {
namespace heir {
namespace preprocessing {

void PreprocessingDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.cpp.inc"
      >();
}

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir
