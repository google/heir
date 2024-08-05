#ifndef LIB_TRANSFORMS_CGGI_BOOLEANLINEVECTORIZER_H_
#define LIB_TRANSFORMS_CGGI_BOOLEANLINEVECTORIZER_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_DECL_BOOLEANLINEVECTORIZER
#include "lib/Dialect/CGGI/Transforms/Passes.h.inc"

}  // namespace cggi
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_CGGI_BOOLEANLINEVECTORIZER_H_
