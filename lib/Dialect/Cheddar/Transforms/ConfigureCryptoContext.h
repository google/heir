#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_H_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

#define GEN_PASS_DECL_CONFIGURECRYPTOCONTEXT
#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h.inc"

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_H_
