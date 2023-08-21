#include "include/Dialect/CGGI/IR/CGGIDialect.h"

#include "include/Dialect/CGGI/IR/CGGIOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// Generated definitions
#include "include/Dialect/CGGI/IR/CGGIDialect.cpp.inc"
#define GET_OP_CLASSES
#include "include/Dialect/CGGI/IR/CGGIOps.cpp.inc"

namespace mlir {
namespace heir {
namespace cggi {

//===----------------------------------------------------------------------===//
// CGGI dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void CGGIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/CGGI/IR/CGGIOps.cpp.inc"
      >();
}

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
