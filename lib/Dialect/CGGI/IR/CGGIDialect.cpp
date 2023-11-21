#include "include/Dialect/CGGI/IR/CGGIDialect.h"

// NOLINTNEXTLINE(misc-include-cleaner): Required to define CGGIOps
#include "include/Dialect/CGGI/IR/CGGIAttributes.h"
#include "include/Dialect/CGGI/IR/CGGIOps.h"

// Generated definitions
#include "include/Dialect/CGGI/IR/CGGIDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "include/Dialect/CGGI/IR/CGGIAttributes.cpp.inc"
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
  addAttributes<
#define GET_ATTRDEF_LIST
#include "include/Dialect/CGGI/IR/CGGIAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/CGGI/IR/CGGIOps.cpp.inc"
      >();

  getContext()->getOrLoadDialect("lwe");
}

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
