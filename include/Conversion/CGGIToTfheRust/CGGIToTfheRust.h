#ifndef INCLUDE_CONVERSION_CGGITOTFHERUST_CGGITOTFHERUST_H_
#define INCLUDE_CONVERSION_CGGITOTFHERUST_CGGITOTFHERUST_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DECL
#include "include/Conversion/CGGIToTfheRust/CGGIToTfheRust.h.inc"

#define GEN_PASS_REGISTRATION
#include "include/Conversion/CGGIToTfheRust/CGGIToTfheRust.h.inc"

}  // namespace mlir::heir

#endif  // INCLUDE_CONVERSION_CGGITOTFHERUST_CGGITOTFHERUST_H_
