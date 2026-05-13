#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"

#include "lib/Dialect/JaxiteWord/IR/JaxiteWordAttributes.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.cpp.inc"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.cpp.inc"

namespace mlir {
namespace heir {
namespace jaxiteword {

void JaxiteWordDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.cpp.inc"
      >();
}

LogicalResult AddOp::verify() {
  if (getModulusList().getType().getModulusList().size() !=
      getValueA().getType().getTowers()) {
    return emitOpError() << "Number of Towers of moudlus should match the "
                            "number of towers/limbs";
  }
  return success();
}

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir
