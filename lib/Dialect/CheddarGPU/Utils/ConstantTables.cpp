#include "lib/Dialect/CheddarGPU/Utils/ConstantTables.h"

#include <cstddef>
#include <cstdint>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

DenseElementsAttr getDevicePrimesConstant(MLIRContext* ctx, ParamsAttr params,
                                          uint32_t word) {
  Type intType = IntegerType::get(ctx, word);
  size_t totalLimbs = params.getNumLimbs();

  SmallVector<APInt> primesVec;
  primesVec.reserve(totalLimbs);
  for (size_t i = 0; i < totalLimbs; ++i) {
    uint64_t q = params.getPrimeAt(i);
    primesVec.push_back(APInt(word, q));
  }

  auto tensorType =
      RankedTensorType::get({static_cast<int64_t>(totalLimbs)}, intType);
  return DenseElementsAttr::get(tensorType, primesVec);
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
