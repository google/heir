#ifndef SCFIRCkks_SCFIRCkksOps_H
#define SCFIRCkks_SCFIRCkksOps_H

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
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
#include "lib/Dialect/SCIFRCkks/IR/SCIFRCkksOps.h.inc"

#endif
