#ifndef LIB_CONVERSION_TOSATOSECRETARITH_TOSATOSECRETARITH_H_
#define LIB_CONVERSION_TOSATOSECRETARITH_TOSATOSECRETARITH_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tosa {

#define GEN_PASS_DECL
#include "lib/Conversion/TosaToSecretArith/TosaToSecretArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/TosaToSecretArith/TosaToSecretArith.h.inc"

}  // namespace tosa
}  // namespace heir
}  // namespace mlir

#endif  // LIB_CONVERSION_TOSATOSECRETARITH_TOSATOSECRETARITH_H_
