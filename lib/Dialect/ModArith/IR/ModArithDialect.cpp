#include "lib/Dialect/ModArith/IR/ModArithDialect.h"

#include <cassert>

// NOLINTBEGIN(misc-include-cleaner): Required to define
// ModArithDialect, ModArithTypes, ModArithOps,
#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner)

// Generated definitions
#include "lib/Dialect/ModArith/IR/ModArithDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"

namespace mlir {
namespace heir {
namespace mod_arith {

void ModArithDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"
      >();
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
