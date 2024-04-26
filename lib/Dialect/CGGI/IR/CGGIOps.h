#ifndef HEIR_LIB_DIALECT_CGGI_IR_CGGIOPS_H_
#define HEIR_LIB_DIALECT_CGGI_IR_CGGIOPS_H_

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/CGGI/IR/CGGIOps.h.inc"

#endif  // HEIR_LIB_DIALECT_CGGI_IR_CGGIOPS_H_
