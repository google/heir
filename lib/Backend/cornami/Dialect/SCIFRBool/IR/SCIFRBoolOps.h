#ifndef SCFIRBool_SCFIRBoolOps_H
#define SCFIRBool_SCFIRBoolOps_H

#include "lib/Dialect/CGGI/IR/CGGIAttributes.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/Bytecode/BytecodeOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Backend/cornami/Dialect/SCIFRBool/IR/SCIFRBoolOps.h.inc"

#endif
