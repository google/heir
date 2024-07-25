#ifndef LIB_CONVERSION_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_H_
#define LIB_CONVERSION_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::lwe {

#define GEN_PASS_DECL
#include "lib/Conversion/LWEToPolynomial/LWEToPolynomial.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/LWEToPolynomial/LWEToPolynomial.h.inc"

}  // namespace mlir::heir::lwe

#endif  // LIB_CONVERSION_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_H_
