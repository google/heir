#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"

#include "lib/Analysis/NoiseAnalysis/BFV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BFV/NoiseByVarianceCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BFV/NoiseCanEmbModel.h"

namespace mlir {
namespace heir {

// template instantiation
template class NoiseAnalysis<bfv::NoiseByBoundCoeffModel>;
template class NoiseAnalysis<bfv::NoiseByVarianceCoeffModel>;
template class NoiseAnalysis<bfv::NoiseCanEmbModel>;

}  // namespace heir
}  // namespace mlir
