//===- CFAIROps.cpp - CFAIR dialect ops ---------------*- C++ -*-===//

#include "lib/Backend/cornami/Dialect/SCIFRBool/IR/SCIFRBoolOps.h"

#include "lib/Backend/cornami/Dialect/SCIFRBool/IR/SCIFRBoolDialect.h"
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::heir;

#define GET_OP_CLASSES
#include "lib/Backend/cornami/Dialect/SCIFRBool/IR/SCIFRBoolOps.cpp.inc"
