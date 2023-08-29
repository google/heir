#ifndef HEIR_INCLUDE_CONVERSION_BGVTOPOLY_BGVTOPOLY_H_
#define HEIR_INCLUDE_CONVERSION_BGVTOPOLY_BGVTOPOLY_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DECL
#include "include/Conversion/BGVToPoly/BGVToPoly.h.inc"

#define GEN_PASS_REGISTRATION
#include "include/Conversion/BGVToPoly/BGVToPoly.h.inc"

}  // namespace mlir::heir::bgv

#endif  // HEIR_INCLUDE_CONVERSION_BGVTOPOLY_BGVTOPOLY_H_
