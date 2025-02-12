#include "lib/Parameters/CKKS/Params.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <numeric>
#include <sstream>
#include <vector>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

SchemeParam SchemeParam::getConcreteSchemeParam(std::vector<double> logqi,
                                                int logDefaultScale) {
  return SchemeParam(
      RLWESchemeParam::getConcreteRLWESchemeParam(std::move(logqi)),
      logDefaultScale);
}

void SchemeParam::print(llvm::raw_ostream &os) const {
  os << "logDefaultScale: " << logDefaultScale << "\n";
  RLWESchemeParam::print(os);
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
