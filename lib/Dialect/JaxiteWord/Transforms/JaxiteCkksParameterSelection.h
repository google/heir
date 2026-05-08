#ifndef LIB_DIALECT_JAXITEWORD_TRANSFORMS_JAXITECKKSPARAMETERSELECTION_H_
#define LIB_DIALECT_JAXITEWORD_TRANSFORMS_JAXITECKKSPARAMETERSELECTION_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace jaxite_word {

#define GEN_PASS_DECL_JAXITECKKSPARAMETERSELECTION
#include "lib/Dialect/JaxiteWord/Transforms/Passes.h.inc"

}  // namespace jaxite_word
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_JAXITEWORD_TRANSFORMS_JAXITECKKSPARAMETERSELECTION_H_
