#include "lib/Dialect/Secret/IR/SecretDialect.h"

// IWYU pragma: begin_keep
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
// IWYU pragma: end_keep

#include "lib/Dialect/Secret/IR/SecretDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Secret/IR/SecretAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Secret/IR/SecretTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/Secret/IR/SecretOps.cpp.inc"

namespace mlir {
namespace heir {
namespace secret {

//===----------------------------------------------------------------------===//
// Secret dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void SecretDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/Secret/IR/SecretAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Secret/IR/SecretTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Secret/IR/SecretOps.cpp.inc"
      >();
}

}  // namespace secret
}  // namespace heir
}  // namespace mlir
