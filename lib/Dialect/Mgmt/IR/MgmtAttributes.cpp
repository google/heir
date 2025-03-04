#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"

#include "lib/Utils/AttributeUtils.h"

namespace mlir {
namespace heir {
namespace mgmt {

MgmtAttr getMgmtAttrFromValue(Value value) {
  return dyn_cast_or_null<MgmtAttr>(
      getAttributeFromValue(value, mgmt::MgmtDialect::kArgMgmtAttrName));
}

void setMgmtAttrForValue(Value value, MgmtAttr mgmtAttr) {
  setAttributeForValue(value, mgmt::MgmtDialect::kArgMgmtAttrName, mgmtAttr);
}

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
