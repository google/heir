#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define LattigoOps

#include "lib/Dialect/Lattigo/IR/LattigoAttributes.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"

// Generated definitions
#include "lib/Dialect/Lattigo/IR/LattigoDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Lattigo/IR/LattigoAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Lattigo/IR/LattigoTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/Lattigo/IR/LattigoOps.cpp.inc"

namespace mlir {
namespace heir {
namespace lattigo {

void LattigoDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/Lattigo/IR/LattigoAttributes.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Lattigo/IR/LattigoTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Lattigo/IR/LattigoOps.cpp.inc"
      >();
}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
