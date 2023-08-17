#include "include/Dialect/BGV/IR/BGVDialect.h"

#include "include/Dialect/BGV/IR/BGVAttributes.h"
#include "include/Dialect/BGV/IR/BGVOps.h"
#include "include/Dialect/BGV/IR/BGVTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// Generated definitions
#include "include/Dialect/BGV/IR/BGVDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "include/Dialect/BGV/IR/BGVAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "include/Dialect/BGV/IR/BGVTypes.cpp.inc"
#define GET_OP_CLASSES
#include "include/Dialect/BGV/IR/BGVOps.cpp.inc"

namespace mlir {
namespace heir {
namespace bgv {

//===----------------------------------------------------------------------===//
// BGV dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void BGVDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "include/Dialect/BGV/IR/BGVAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "include/Dialect/BGV/IR/BGVTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/BGV/IR/BGVOps.cpp.inc"
      >();
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
