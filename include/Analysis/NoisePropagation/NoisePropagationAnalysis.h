#ifndef INCLUDE_ANALYSIS_NOISEPROPAGATION_NOISEPROPAGATIONANALYSIS_H_
#define INCLUDE_ANALYSIS_NOISEPROPAGATION_NOISEPROPAGATIONANALYSIS_H_

#include "include/Analysis/NoisePropagation/Variance.h"
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project

namespace mlir {
namespace heir {

/// This lattice element represents the noise distribution of an SSA value.
class VarianceLattice : public dataflow::Lattice<Variance> {
 public:
  using Lattice::Lattice;
};

/// Noise propagation analysis determines a noise bound for SSA values,
/// represented by the variance of a symmetric Gaussian distribution. This
/// analysis propagates noise across operations that implement
/// `NoisePropagationInterface`, but does not support propagation for SSA
/// values that represent loop bounds or induction variables. It can be viewed
/// as a simplified port of IntegerRangeAnalysis.
class NoisePropagationAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<VarianceLattice> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void setToEntryState(VarianceLattice *lattice) override {
    // At an entry point, we have no information about the noise.
    propagateIfChanged(lattice, lattice->join(Variance(std::nullopt)));
  }

  void visitOperation(Operation *op, ArrayRef<const VarianceLattice *> operands,
                      ArrayRef<VarianceLattice *> results) override;
};

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEPROPAGATION_NOISEPROPAGATIONANALYSIS_H_
