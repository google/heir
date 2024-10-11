#ifndef LIB_DIALECT_CGGI_CONVERSIONS_CGGITOTFHERUST_CGGITOTFHERUST_H_
#define LIB_DIALECT_CGGI_CONVERSIONS_CGGITOTFHERUST_CGGITOTFHERUST_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DECL
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRust/CGGIToTfheRust.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRust/CGGIToTfheRust.h.inc"

}  // namespace mlir::heir

#endif  // LIB_DIALECT_CGGI_CONVERSIONS_CGGITOTFHERUST_CGGITOTFHERUST_H_
