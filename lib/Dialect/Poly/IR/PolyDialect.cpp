#include "include/Dialect/Poly/IR/PolyDialect.h"
#include "include/Dialect/Poly/IR/PolyTypes.h"
#include "include/Dialect/Poly/IR/PolyOps.h"

#include "llvm/include/llvm/ADT/TypeSwitch.h" // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h" // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h" // from @llvm-project

#include "include/Dialect/Poly/IR/PolyDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "include/Dialect/Poly/IR/PolyTypes.cpp.inc"
#define GET_OP_CLASSES
#include "include/Dialect/Poly/IR/PolyOps.cpp.inc"

namespace mlir {
namespace heir {
namespace poly {

//===----------------------------------------------------------------------===//
// Poly dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void PolyDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "include/Dialect/Poly/IR/PolyTypes.cpp.inc"
      >();
    addOperations<
#define GET_OP_LIST
#include "include/Dialect/Poly/IR/PolyOps.cpp.inc"
      >();
}

}  // namespace poly
}  // namespace heir
}  // namespace mlir

