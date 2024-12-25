#ifndef LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGI_ARITHTOCGGI_H_
#define LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGI_ARITHTOCGGI_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::arith {

#define GEN_PASS_DECL
#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h.inc"

}  // namespace mlir::heir::arith

#endif  // LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGI_ARITHTOCGGI_H_
