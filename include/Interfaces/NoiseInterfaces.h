#ifndef INCLUDE_INTERFACES_NOISEINTERFACES_H_
#define INCLUDE_INTERFACES_NOISEINTERFACES_H_

#include "include/Analysis/NoisePropagation/Variance.h"
#include "mlir/include/mlir/IR/OpDefinition.h"  // trom @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project

namespace mlir {
namespace heir {

// Variance is a type defined by NoisePropagationAnalysis
using SetNoiseFn = function_ref<void(Value, Variance)>;

}  // namespace heir
}  // namespace mlir

#include "include/Interfaces/NoiseInterfaces.h.inc"

#endif  // INCLUDE_INTERFACES_NOISEINTERFACES_H_
