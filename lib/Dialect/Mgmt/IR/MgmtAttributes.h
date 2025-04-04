#ifndef LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_H_
#define LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_H_

#include <cstdint>

#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "mlir/include/mlir/IR/Value.h"  // from @llvm-project

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h.inc"

namespace mlir {
namespace heir {
namespace mgmt {

//===----------------------------------------------------------------------===//
// MgmtAttr helpers
//===----------------------------------------------------------------------===//

MgmtAttr getMgmtAttrWithNewScale(MgmtAttr mgmtAttr, int64_t scale);

//===----------------------------------------------------------------------===//
// Getters and Setters
//===----------------------------------------------------------------------===//

/// find the MgmtAttr associated with the given Value
MgmtAttr findMgmtAttrAssociatedWith(Value value);

/// set the MgmtAttr associated with the given Value
void setMgmtAttrAssociatedWith(Value value, MgmtAttr mgmtAttr);

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_H_
