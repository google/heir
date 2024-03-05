#ifndef INCLUDE_CONVERSION_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_H_
#define INCLUDE_CONVERSION_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DECL
#include "include/Conversion/CGGIToTfheRustBool/CGGIToTfheRustBool.h.inc"

#define GEN_PASS_REGISTRATION
#include "include/Conversion/CGGIToTfheRustBool/CGGIToTfheRustBool.h.inc"

}  // namespace mlir::heir

#endif  // INCLUDE_CONVERSION_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_H_
