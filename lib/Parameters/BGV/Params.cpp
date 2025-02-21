#include "lib/Parameters/BGV/Params.h"

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Parameters/RLWEParams.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace bgv {

SchemeParam SchemeParam::getConservativeSchemeParam(int level,
                                                    int64_t plaintextModulus) {
  return SchemeParam(RLWESchemeParam::getConservativeRLWESchemeParam(level),
                     plaintextModulus);
}

SchemeParam SchemeParam::getConcreteSchemeParam(std::vector<double> logqi,
                                                int64_t plaintextModulus) {
  return SchemeParam(RLWESchemeParam::getConcreteRLWESchemeParam(
                         std::move(logqi), plaintextModulus),
                     plaintextModulus);
}

void SchemeParam::print(llvm::raw_ostream &os) const {
  os << "plaintextModulus: " << plaintextModulus << "\n";
  RLWESchemeParam::print(os);
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
