#include "lib/Analysis/NoiseAnalysis/Params.h"

#include <cassert>

namespace mlir {
namespace heir {

struct RLWEParam {
  int ringDim;
  int maxQ;
};

// uniform tenary
static struct RLWEParam HEStd_128_classic[] = {
    {1024, 27},   {2048, 54},   {4096, 109},   {8192, 218},
    {16384, 438}, {32768, 881}, {65536, 1747}, {131072, 3523}};

// from OpenFHE
int computeDnum(int level) {
  if (level > 3) {
    return 3;
  }
  if (level > 0) {
    return 2;
  }
  return 1;
}

SchemeParam SchemeParam::getConservativeSchemeParam(int level,
                                                    int64_t plaintextModulus) {
  auto logModuli = 55;  // assume 55 bit moduli
  auto dnum = computeDnum(level);
  std::vector<int> logqi(level + 1, logModuli);
  std::vector<int> logpi(ceil(static_cast<double>(level) / dnum), logModuli);

  auto totalQP = logModuli * (logqi.size() + logpi.size());

  auto ringDim = 0;
  for (auto &param : HEStd_128_classic) {
    if (param.maxQ >= totalQP) {
      ringDim = param.ringDim;
      break;
    }
  }
  assert(ringDim != 0 && "Failed to find ring dimension, level too high");

  return SchemeParam(ringDim, plaintextModulus, level, logqi, dnum, logpi);
}

}  // namespace heir
}  // namespace mlir
