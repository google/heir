#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"

#include <cstdint>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

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

bool shouldHaveMgmtAttribute(Value value, DataFlowSolver* solver) {
  return isSecret(value, solver) ||
         (isa_and_nonnull<mgmt::InitOp>(value.getDefiningOp()));
}

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
