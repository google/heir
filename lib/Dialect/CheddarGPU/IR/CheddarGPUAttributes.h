#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ATTRIBUTES_H_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ATTRIBUTES_H_

#include <cstddef>
#include <cstdint>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "mlir/include/mlir/IR/Attributes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h.inc"

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ATTRIBUTES_H_
