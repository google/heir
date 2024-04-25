#include "include/Dialect/Jaxite/IR/JaxiteDialect.h"

#include "include/Dialect/Jaxite/IR/JaxiteDialect.cpp.inc"
#include "include/Dialect/Jaxite/IR/JaxiteOps.h"
#include "include/Dialect/Jaxite/IR/JaxiteTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#define GET_TYPEDEF_CLASSES
#include "include/Dialect/Jaxite/IR/JaxiteTypes.cpp.inc"
#define GET_OP_CLASSES
#include "include/Dialect/Jaxite/IR/JaxiteOps.cpp.inc"

namespace mlir {
namespace heir {
namespace jaxite {

void JaxiteDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "include/Dialect/Jaxite/IR/JaxiteTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/Jaxite/IR/JaxiteOps.cpp.inc"
      >();
}

}  // namespace jaxite
}  // namespace heir
}  // namespace mlir
