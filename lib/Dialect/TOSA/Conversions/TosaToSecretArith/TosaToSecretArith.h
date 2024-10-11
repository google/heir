#ifndef LIB_DIALECT_TOSA_CONVERSIONS_TOSATOSECRETARITH_TOSATOSECRETARITH_H_
#define LIB_DIALECT_TOSA_CONVERSIONS_TOSATOSECRETARITH_TOSATOSECRETARITH_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tosa {

#define GEN_PASS_DECL
#include "lib/Dialect/TOSA/Conversions/TosaToSecretArith/TosaToSecretArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/TOSA/Conversions/TosaToSecretArith/TosaToSecretArith.h.inc"

}  // namespace tosa
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_TOSA_CONVERSIONS_TOSATOSECRETARITH_TOSATOSECRETARITH_H_
