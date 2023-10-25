#ifndef INCLUDE_CONVERSION_BGVTOPOLYNOMIAL_BGVTOPOLYNOMIAL_H_
#define INCLUDE_CONVERSION_BGVTOPOLYNOMIAL_BGVTOPOLYNOMIAL_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DECL
#include "include/Conversion/BGVToPolynomial/BGVToPolynomial.h.inc"

#define GEN_PASS_REGISTRATION
#include "include/Conversion/BGVToPolynomial/BGVToPolynomial.h.inc"

}  // namespace mlir::heir::bgv

#endif  // INCLUDE_CONVERSION_BGVTOPOLYNOMIAL_BGVTOPOLYNOMIAL_H_
