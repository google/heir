#include "include/Dialect/Polynomial/IR/PolynomialDialect.h"

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define PolynomialOps
#include "include/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "include/Dialect/Polynomial/IR/PolynomialDetail.h"
#include "include/Dialect/Polynomial/IR/PolynomialOps.h"
#include "include/Dialect/Polynomial/IR/PolynomialTypes.h"

// Generated definitions
#include "include/Dialect/Polynomial/IR/PolynomialDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "include/Dialect/Polynomial/IR/PolynomialAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "include/Dialect/Polynomial/IR/PolynomialTypes.cpp.inc"
#define GET_OP_CLASSES
#include "include/Dialect/Polynomial/IR/PolynomialOps.cpp.inc"

namespace mlir {
namespace heir {
namespace polynomial {

//===----------------------------------------------------------------------===//
// Poly dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void PolynomialDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "include/Dialect/Polynomial/IR/PolynomialAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "include/Dialect/Polynomial/IR/PolynomialTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/Polynomial/IR/PolynomialOps.cpp.inc"
      >();

  getContext()
      ->getAttributeUniquer()
      .registerParametricStorageType<detail::PolynomialStorage>();

  // Needed for canonicalization patterns that create new arith ops
  getContext()->getOrLoadDialect("arith");
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
