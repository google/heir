#ifndef LIB_PARAMETERS_CKKS_UTILS_H_
#define LIB_PARAMETERS_CKKS_UTILS_H_

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Parameters/CKKS/Params.h"

namespace mlir {
namespace heir {
namespace ckks {

SchemeParam getSchemeParamFromAttr(SchemeParamAttr attr);

}
}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_CKKS_UTILS_H_
