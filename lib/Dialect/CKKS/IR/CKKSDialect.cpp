#include "lib/Dialect/CKKS/IR/CKKSDialect.h"

// IWYU pragma: begin_keep
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
// IWYU pragma: end_keep

// Generated definitions
#include "lib/Dialect/CKKS/IR/CKKSDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/CKKS/IR/CKKSAttributes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/CKKS/IR/CKKSOps.cpp.inc"

namespace mlir {
namespace heir {
namespace ckks {

//===----------------------------------------------------------------------===//
// CKKS dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void CKKSDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/CKKS/IR/CKKSAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/CKKS/IR/CKKSOps.cpp.inc"
      >();
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
