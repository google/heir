#ifndef LIB_TRANSFORMS_PROPAGATEANNOTATION_PROPAGATEANNOTATION_H_
#define LIB_TRANSFORMS_PROPAGATEANNOTATION_PROPAGATEANNOTATION_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h.inc"

/// Forward propagate an annotation, with a callable determining when it's
/// OK to propagate based on the type.
void forwardPropagateAnnotation(Operation *root, StringRef attrName,
                                function_ref<bool(Type)> shouldPropagate);

/// Forward-propagate an annotation through the IR.
inline void forwardPropagateAnnotation(Operation *root, StringRef attrName) {
  return forwardPropagateAnnotation(root, attrName, [](Type) { return true; });
}

/// Backward propagate an annotation, with a callable determining when it's
/// OK to propagate based on the type.
void backwardPropagateAnnotation(Operation *root, StringRef attrName,
                                 function_ref<bool(Type)> shouldPropagate);

/// Backward-propagate an annotation through the IR.
inline void backwardPropagateAnnotation(Operation *root, StringRef attrName) {
  return backwardPropagateAnnotation(root, attrName, [](Type) { return true; });
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_PROPAGATEANNOTATION_PROPAGATEANNOTATION_H_
