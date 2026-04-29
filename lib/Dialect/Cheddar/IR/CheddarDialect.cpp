#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define CheddarOps

#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"

// Generated definitions
#include "lib/Dialect/Cheddar/IR/CheddarDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarOps.cpp.inc"

namespace mlir {
namespace heir {
namespace cheddar {

void CheddarDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Cheddar/IR/CheddarTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Cheddar/IR/CheddarOps.cpp.inc"
      >();
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
