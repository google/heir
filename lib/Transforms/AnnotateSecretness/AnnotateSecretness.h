#ifndef LIB_TRANSFORMS_ANNOTATESECRETNESS_ANNOTATESECRETNESS_H_
#define LIB_TRANSFORMS_ANNOTATESECRETNESS_ANNOTATESECRETNESS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/AnnotateSecretness/AnnotateSecretness.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/AnnotateSecretness/AnnotateSecretness.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_ANNOTATESECRETNESS_ANNOTATESECRETNESS_H_
