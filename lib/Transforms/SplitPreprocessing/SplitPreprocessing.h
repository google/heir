#ifndef LIB_TRANSFORMS_SPLITPREPROCESSING_SPLITPREPROCESSING_H_
#define LIB_TRANSFORMS_SPLITPREPROCESSING_SPLITPREPROCESSING_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL_SPLITPREPROCESSING
#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_SPLITPREPROCESSING_SPLITPREPROCESSING_H_
