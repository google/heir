#ifndef INCLUDE_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_H_
#define INCLUDE_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "include/Transforms/UnusedMemRef/UnusedMemRef.h.inc"

#define GEN_PASS_REGISTRATION
#include "include/Transforms/UnusedMemRef/UnusedMemRef.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_H_
