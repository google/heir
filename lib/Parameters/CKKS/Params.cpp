#include "lib/Parameters/CKKS/Params.h"

#include <cassert>
#include <utility>
#include <vector>

#include "lib/Parameters/RLWEParams.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

SchemeParam SchemeParam::getConcreteSchemeParam(std::vector<double> logqi,
                                                int logDefaultScale,
                                                int slotNumber) {
  // CKKS slot number = ringDim / 2
  return SchemeParam(RLWESchemeParam::getConcreteRLWESchemeParam(
                         std::move(logqi), 2 * slotNumber),
                     logDefaultScale);
}

void SchemeParam::print(llvm::raw_ostream &os) const {
  os << "logDefaultScale: " << logDefaultScale << "\n";
  RLWESchemeParam::print(os);
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
