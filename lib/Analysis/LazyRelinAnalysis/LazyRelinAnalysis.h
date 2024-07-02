#ifndef LIB_ANALYSIS_LAZYRELINANALYSIS_H
#define LIB_ANALYSIS_LAZYRELINANALYSIS_H

#include "llvm/include/llvm/ADT/DenseMap.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Value.h"
namespace mlir{
namespace heir{
    class LazyRelinAnalysis{
        public:
            LazyRelinAnalysis(Operation *op);
            ~LazyRelinAnalysis() = default;

            // Return true if a relin op should be inserted after the given
            // operation, according to the solution to the optimization problem.
            bool shouldInsertRelin(Operation *op)  const{
                return solution.lookup(op);
            }

        private:
            llvm::DenseMap<Operation *, bool> solution;

    };
}  // namespace heir
}  // namespace mlir



#endif   // LIB_ANALYSIS_LAZYRELINANALYSIS_H