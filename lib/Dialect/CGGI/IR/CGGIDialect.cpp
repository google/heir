#include "lib/Dialect/CGGI/IR/CGGIDialect.h"

// NOLINTNEXTLINE(misc-include-cleaner): Required to define CGGIOps
#include "lib/Dialect/CGGI/IR/CGGIAttributes.h"
#include "lib/Dialect/CGGI/IR/CGGIEnums.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// Generated definitions
#include "lib/Dialect/CGGI/IR/CGGIDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/CGGI/IR/CGGIAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/CGGI/IR/CGGIEnums.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/CGGI/IR/CGGIOps.cpp.inc"

namespace mlir {
namespace heir {
namespace cggi {

//===----------------------------------------------------------------------===//
// CGGI dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void CGGIDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/CGGI/IR/CGGIAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/CGGI/IR/CGGIOps.cpp.inc"
      >();

  getContext()->getOrLoadDialect("lwe");
}
}  // namespace cggi
}  // namespace heir
}  // namespace mlir
