#include "lib/Dialect/RNS/IR/RNSDialect.h"

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define RNSOps
#include "lib/Dialect/RNS/IR/RNSAttributes.h"
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"

// Generated definitions
#include "lib/Dialect/RNS/IR/RNSDialect.cpp.inc"
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"         // from @llvm-project
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/RNS/IR/RNSAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/RNS/IR/RNSTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/RNS/IR/RNSOps.cpp.inc"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.cpp.inc"

namespace mlir {
namespace heir {
namespace rns {

void RNSDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/RNS/IR/RNSAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/RNS/IR/RNSTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/RNS/IR/RNSOps.cpp.inc"
      >();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
