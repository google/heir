#ifndef INCLUDE_TRANSFORMS_APPLYFOLDERS_APPLYFOLDERS_H_
#define INCLUDE_TRANSFORMS_APPLYFOLDERS_APPLYFOLDERS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "include/Transforms/ApplyFolders/ApplyFolders.h.inc"

#define GEN_PASS_REGISTRATION
#include "include/Transforms/ApplyFolders/ApplyFolders.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_TRANSFORMS_APPLYFOLDERS_APPLYFOLDERS_H_
