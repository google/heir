#include "include/Dialect/RNS/IR/RNSDialect.h"

#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define RNSOps

#include "include/Dialect/RNS/IR/RNSOps.h"
#include "include/Dialect/RNS/IR/RNSTypes.h"

// Generated definitions
#include "include/Dialect/RNS/IR/RNSDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "include/Dialect/RNS/IR/RNSTypes.cpp.inc"

#define GET_OP_CLASSES
#include "include/Dialect/RNS/IR/RNSOps.cpp.inc"

namespace mlir {
namespace heir {
namespace rns {

void RNSDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "include/Dialect/RNS/IR/RNSTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "include/Dialect/RNS/IR/RNSOps.cpp.inc"
      >();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
