#ifndef LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_H_
#define LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_H_

#include <cstdint>

#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h.inc"

namespace mlir {
namespace heir {
namespace mgmt {

//===----------------------------------------------------------------------===//
// MgmtAttr helpers
//===----------------------------------------------------------------------===//

MgmtAttr getMgmtAttrWithNewScale(MgmtAttr mgmtAttr, const llvm::APInt& scale);
MgmtAttr getMgmtAttrWithNewScale(MgmtAttr mgmtAttr, int64_t scale);

//===----------------------------------------------------------------------===//
// Getters and Setters
//===----------------------------------------------------------------------===//

/// find the MgmtAttr associated with the given Value
MgmtAttr findMgmtAttrAssociatedWith(Value value);

/// set the MgmtAttr associated with the given Value
void setMgmtAttrAssociatedWith(Value value, MgmtAttr mgmtAttr);

/// Determine if a Value should have an associated mgmt attribute. This is
/// intended to be used by various analyses that populate attributes to be used
/// by later conversion passes, and it includes secret values (which become
/// ciphertexts) as well as some non-secret values (e.g., the result of a
/// mgmt.init op) which become plaintexts with specific encoding details.
bool shouldHaveMgmtAttribute(Value value, DataFlowSolver* solver);

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_H_
