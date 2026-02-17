#include "lib/Dialect/KeyMgmt/IR/KeyMgmtDialect.h"

#include "lib/Dialect/KeyMgmt/IR/KeyMgmtDialect.cpp.inc"
#include "lib/Dialect/KeyMgmt/IR/KeyMgmtOps.h"
#include "lib/Dialect/KeyMgmt/IR/KeyMgmtTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/KeyMgmt/IR/KeyMgmtTypes.cpp.inc"

namespace mlir {
namespace heir {
namespace key_mgmt {

void KeyMgmtDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/KeyMgmt/IR/KeyMgmtTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/KeyMgmt/IR/KeyMgmtOps.cpp.inc"
      >();
}

}  // namespace key_mgmt
}  // namespace heir
}  // namespace mlir
