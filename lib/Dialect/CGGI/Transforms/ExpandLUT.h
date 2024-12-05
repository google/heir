#ifndef LIB_TRANSFORMS_CGGI_EXPANDLUT_H_
#define LIB_TRANSFORMS_CGGI_EXPANDLUT_H_

#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_DECL_EXPANDLUT
#include "lib/Dialect/CGGI/Transforms/Passes.h.inc"

}  // namespace cggi
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_CGGI_EXPANDLUT_H_
