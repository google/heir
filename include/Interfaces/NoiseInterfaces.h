#ifndef INCLUDE_INTERFACES_NOISEINTERFACES_H_
#define INCLUDE_INTERFACES_NOISEINTERFACES_H_

#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project

namespace mlir {
namespace heir {

using SetNoiseFn = function_ref<void(Value, int64_t)>;

}
}  // namespace mlir

#include "include/Interfaces/NoiseInterfaces.h.inc"

#endif  // INCLUDE_INTERFACES_NOISEINTERFACES_H_
