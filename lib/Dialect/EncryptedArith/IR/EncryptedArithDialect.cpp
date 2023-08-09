#include "include/Dialect/EncryptedArith/IR/EncryptedArithDialect.h"

#include "include/Dialect/EncryptedArith/IR/EncryptedArithTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// Generated definitions.
#include "include/Dialect/EncryptedArith/IR/EncryptedArithDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "include/Dialect/EncryptedArith/IR/EncryptedArithTypes.cpp.inc"

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// EncryptedArith dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void EncryptedArithDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "include/Dialect/EncryptedArith/IR/EncryptedArithTypes.cpp.inc"
      >();
}

}  // namespace heir
}  // namespace mlir
