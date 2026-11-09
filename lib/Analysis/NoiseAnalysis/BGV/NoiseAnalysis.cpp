#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"

#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByVarianceCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseCanEmbModel.h"

namespace mlir {
namespace heir {

// template instantiation
template class NoiseAnalysis<bgv::NoiseByBoundCoeffModel>;
template class NoiseAnalysis<bgv::NoiseCanEmbModel>;
template class NoiseAnalysis<bgv::NoiseByVarianceCoeffModel>;

}  // namespace heir
}  // namespace mlir
