#ifndef THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_H_
#define THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_H_

#include "mlir/include/mlir/Pass/Pass.h" // from @llvm-project

namespace mlir {

namespace heir {

std::unique_ptr<Pass> createMemrefGlobalReplacePass();

std::unique_ptr<Pass> createExpandCopyPass();

std::unique_ptr<Pass> createUnrollAndForwardStoresPass();

std::unique_ptr<Pass> createExtractLoopBodyPass();

}  // namespace heir

}  // namespace mlir

#endif  // THIRD_PARTY_HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_H_
