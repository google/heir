#include "include/Dialect/TfheRust/IR/TfheRustDialect.h"

#include "include/Dialect/TfheRust/IR/TfheRustDialect.cpp.inc"
#include "include/Dialect/TfheRust/IR/TfheRustOps.h"
#include "include/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#define GET_TYPEDEF_CLASSES
#include "include/Dialect/TfheRust/IR/TfheRustTypes.cpp.inc"
#define GET_OP_CLASSES
#include "include/Dialect/TfheRust/IR/TfheRustOps.cpp.inc"

namespace mlir {
namespace heir {
namespace tfhe_rust {

//===----------------------------------------------------------------------===//
// TfheRust dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void TfheRustDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "include/Dialect/TfheRust/IR/TfheRustTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/TfheRust/IR/TfheRustOps.cpp.inc"
      >();
}

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir
