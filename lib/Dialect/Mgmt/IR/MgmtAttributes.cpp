#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"

#include "lib/Utils/AttributeUtils.h"

namespace mlir {
namespace heir {
namespace mgmt {

MgmtAttr findMgmtAttrAssociatedWith(Value value) {
  return dyn_cast_or_null<MgmtAttr>(
      findAttributeAssociatedWith(value, mgmt::MgmtDialect::kArgMgmtAttrName));
}
//
// void setMgmtAttrForValue(Value value, MgmtAttr mgmtAttr) {
//  setAttributeForValue(value, mgmt::MgmtDialect::kArgMgmtAttrName, mgmtAttr);
//}

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
