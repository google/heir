#include "lib/Dialect/Orion/IR/OrionDialect.h"

#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define OrionOps

#include "lib/Dialect/Orion/IR/OrionOps.h"

// Generated definitions
#include "lib/Dialect/Orion/IR/OrionDialect.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/Orion/IR/OrionOps.cpp.inc"
