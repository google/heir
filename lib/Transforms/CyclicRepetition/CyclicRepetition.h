#ifndef LIB_TRANSFORMS_CYCLICREPETITION_CYCLICREPETITION_H_
#define LIB_TRANSFORMS_CYCLICREPETITION_CYCLICREPETITION_H_

#include <memory>

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL_CYCLICREPETITION
#include "lib/Transforms/CyclicRepetition/CyclicRepetition.h.inc"

std::unique_ptr<Pass> createCyclicRepetition();

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_CYCLICREPETITION_CYCLICREPETITION_H_
