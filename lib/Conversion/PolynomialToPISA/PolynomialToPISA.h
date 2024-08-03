#ifndef LIB_CONVERSION_POLYNOMIALTOPISA_POLYNOMIALTOPISA_H_
#define LIB_CONVERSION_POLYNOMIALTOPISA_POLYNOMIALTOPISA_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DECL
#include "lib/Conversion/PolynomialToPISA/PolynomialToPISA.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/PolynomialToPISA/PolynomialToPISA.h.inc"

}  // namespace mlir::heir

#endif  // LIB_CONVERSION_POLYNOMIALTOPISA_POLYNOMIALTOPISA_H_
