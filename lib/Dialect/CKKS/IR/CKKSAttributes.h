#ifndef LIB_DIALECT_CKKS_IR_CKKSATTRIBUTES_H_
#define LIB_DIALECT_CKKS_IR_CKKSATTRIBUTES_H_

#include "lib/Parameters/CKKS/Params.h"

// IWYU pragma: begin_keep
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
// IWYU pragma: end_keep

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h.inc"

namespace mlir {
namespace heir {
namespace ckks {

SchemeParam getSchemeParamFromAttr(SchemeParamAttr attr);

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CKKS_IR_CKKSATTRIBUTES_H_
