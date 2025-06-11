#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"

#include <cstdint>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

// NOLINTBEGIN(misc-include-cleaner): Required to define TensorExt
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner): Required to define TensorExt

// Generated definitions
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/TensorExt/IR/TensorExtOps.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.cpp.inc"

namespace mlir {
namespace heir {
namespace tensor_ext {

void TensorExtDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/TensorExt/IR/TensorExtOps.cpp.inc"
      >();

  // FIXME: determine a better place to put these!
  // declarePromisedInterface<LayoutConversionHoistableOpInterface,
  // arith::AddIOp>();
  // declarePromisedInterface<LayoutConversionHoistableOpInterface,
  // arith::AddFOp>();
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
