#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define MgmtOps

#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"

// Generated definitions
#include "lib/Dialect/Mgmt/IR/MgmtDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/Mgmt/IR/MgmtOps.cpp.inc"

namespace mlir {
namespace heir {
namespace mgmt {

void MgmtDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Mgmt/IR/MgmtOps.cpp.inc"
      >();
}

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
