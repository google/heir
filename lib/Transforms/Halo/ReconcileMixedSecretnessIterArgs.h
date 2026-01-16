#ifndef LIB_TRANSFORMS_HALO_RECONCILEMIXEDSECRETNESSITERARGS_H_
#define LIB_TRANSFORMS_HALO_RECONCILEMIXEDSECRETNESSITERARGS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL_RECONCILEMIXEDSECRETNESSITERARGS
#include "lib/Transforms/Halo/Halo.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_HALO_RECONCILEMIXEDSECRETNESSITERARGS_H_
