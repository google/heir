#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"

#include "lib/Utils/AttributeUtils.h"

namespace mlir {
namespace heir {
namespace mgmt {

MgmtAttr findMgmtAttrAssociatedWith(Value value) {
  auto attr =
      findAttributeAssociatedWith(value, mgmt::MgmtDialect::kArgMgmtAttrName);
  if (failed(attr)) {
    return nullptr;
  }
  return dyn_cast<MgmtAttr>(*attr);
}
//
// void setMgmtAttrForValue(Value value, MgmtAttr mgmtAttr) {
//  setAttributeForValue(value, mgmt::MgmtDialect::kArgMgmtAttrName, mgmtAttr);
//}

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
