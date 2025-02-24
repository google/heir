#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"

#include "lib/Utils/AttributeUtils.h"

namespace mlir {
namespace heir {
namespace mgmt {

//===----------------------------------------------------------------------===//
// MgmtAttr helpers
//===----------------------------------------------------------------------===//

MgmtAttr getMgmtAttrWithNewScale(MgmtAttr mgmtAttr, int64_t scale) {
  return MgmtAttr::get(mgmtAttr.getContext(), mgmtAttr.getLevel(),
                       mgmtAttr.getDimension(), scale);
}

//===----------------------------------------------------------------------===//
// Getters and Setters
//===----------------------------------------------------------------------===//

MgmtAttr findMgmtAttrAssociatedWith(Value value) {
  auto attr =
      findAttributeAssociatedWith(value, mgmt::MgmtDialect::kArgMgmtAttrName);
  if (failed(attr)) {
    return nullptr;
  }
  return dyn_cast<MgmtAttr>(*attr);
}

void setMgmtAttrAssociatedWith(Value value, MgmtAttr mgmtAttr) {
  setAttributeAssociatedWith(value, mgmt::MgmtDialect::kArgMgmtAttrName,
                             mgmtAttr);
}

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
