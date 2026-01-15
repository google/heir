#ifndef LIB_TRANSFORMS_BOOLEANVECTORIZER_BOOLEANVECTORIZER_H_
#define LIB_TRANSFORMS_BOOLEANVECTORIZER_BOOLEANVECTORIZER_H_

#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"           // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/BooleanVectorizer/BooleanVectorizer.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/BooleanVectorizer/BooleanVectorizer.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_BOOLEANVECTORIZER_BOOLEANVECTORIZER_H_
