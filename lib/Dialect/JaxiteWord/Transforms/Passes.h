#ifndef LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_H_

#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"
#include "lib/Dialect/JaxiteWord/Transforms/JaxiteCkksParameterSelection.h"

namespace mlir {
namespace heir {
namespace jaxite_word {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/JaxiteWord/Transforms/Passes.h.inc"

}  // namespace jaxite_word
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_H_
