#ifndef LIB_CONVERSION_BGVTOPOLYNOMIAL_BGVTOPOLYNOMIAL_H_
#define LIB_CONVERSION_BGVTOPOLYNOMIAL_BGVTOPOLYNOMIAL_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DECL
#include "lib/Conversion/BGVToPolynomial/BGVToPolynomial.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/BGVToPolynomial/BGVToPolynomial.h.inc"

}  // namespace mlir::heir::bgv

#endif  // LIB_CONVERSION_BGVTOPOLYNOMIAL_BGVTOPOLYNOMIAL_H_
