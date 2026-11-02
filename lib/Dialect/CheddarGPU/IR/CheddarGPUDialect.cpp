#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.h"

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.cpp.inc"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.cpp.inc"

// Generated headers
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.cpp.inc"

namespace mlir {
namespace heir {
namespace cheddar_gpu {

void CheddarGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.cpp.inc"
      >();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
