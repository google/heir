//===- CFAIROps.cpp - CFAIR dialect ops ---------------*- C++ -*-===//

#include "lib/Dialect/SCIFRCkks/IR/SCIFRCkksOps.h"

#include "lib/Dialect/SCIFRCkks/IR/SCIFRCkksDialect.h"
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::heir;

#define GET_OP_CLASSES
#include "lib/Dialect/SCIFRCkks/IR/SCIFRCkksOps.cpp.inc"
