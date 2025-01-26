#include "lib/Parameters/BGV/Params.h"

#include <cassert>
#include <iomanip>
#include <sstream>

namespace mlir {
namespace heir {
namespace bgv {

// struct for recording the maximal Q for each ring dim
// under certain security condition.
struct RLWESecurityParam {
  int ringDim;
  int logMaxQ;
};

// 128-bit classic security for uniform ternary secret distribution
// taken from the "Homomorphic Encryption Standard" Preprint
// https://ia.cr/2019/939
// logMaxQ for 65536/131072 taken from OpenFHE
// https://github.com/openfheorg/openfhe-development/blob/7b8346f4eac27121543e36c17237b919e03ec058/src/core/lib/lattice/stdlatticeparms.cpp#L187
static struct RLWESecurityParam HEStd_128_classic[] = {
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

int computeRingDim(int logTotalPQ) {
  for (auto &param : HEStd_128_classic) {
    if (param.logMaxQ >= logTotalPQ) {
      return param.ringDim;
    }
  }
  assert(false && "Failed to find ring dimension, level too large");
  return 0;
}

SchemeParam SchemeParam::getConservativeSchemeParam(int level,
                                                    int64_t plaintextModulus) {
  auto logModuli = 60;  // assume all 60 bit moduli
  auto dnum = computeDnum(level);
  std::vector<double> logqi(level + 1, logModuli);
  std::vector<double> logpi(ceil(static_cast<double>(logqi.size()) / dnum),
                            logModuli);

  auto totalQP = logModuli * (logqi.size() + logpi.size());

  auto ringDim = computeRingDim(totalQP);

  return SchemeParam(ringDim, plaintextModulus, level, logqi, dnum, logpi);
}

void SchemeParam::print(llvm::raw_ostream &os) const {
  auto doubleToString = [](double d) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << d;
    return stream.str();
  };

  os << "ringDim: " << ringDim << "\n";
  os << "plaintextModulus: " << plaintextModulus << "\n";
  os << "level: " << level << "\n";
  os << "logqi: ";
  for (auto qi : logqi) {
    os << doubleToString(qi) << " ";
  }
  os << "\n";
  os << "dnum: " << dnum << "\n";
  os << "logpi: ";
  for (auto pi : logpi) {
    os << doubleToString(pi) << " ";
  }
  os << "\n";
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
