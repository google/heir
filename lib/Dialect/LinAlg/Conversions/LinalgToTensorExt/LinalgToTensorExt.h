#ifndef LIB_DIALECT_LINALG_CONVERSIONS_LINALGTOTENSOREXT_LINALGTOTENSOREXT_H_
#define LIB_DIALECT_LINALG_CONVERSIONS_LINALGTOTENSOREXT_LINALGTOTENSOREXT_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace linalg {

#define GEN_PASS_DECL
#include "lib/Dialect/LinAlg/Conversions/LinalgToTensorExt/LinalgToTensorExt.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/LinAlg/Conversions/LinalgToTensorExt/LinalgToTensorExt.h.inc"

}  // namespace linalg
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LINALG_CONVERSIONS_LINALGTOTENSOREXT_LINALGTOTENSOREXT_H_
