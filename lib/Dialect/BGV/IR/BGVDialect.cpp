#include "include/Dialect/BGV/IR/BGVDialect.h"

#include "include/Dialect/BGV/IR/BGVAttributes.h"
#include "include/Dialect/BGV/IR/BGVOps.h"
#include "include/Dialect/BGV/IR/BGVTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// Generated definitions
#include "include/Dialect/BGV/IR/BGVDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "include/Dialect/BGV/IR/BGVAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "include/Dialect/BGV/IR/BGVTypes.cpp.inc"
#define GET_OP_CLASSES
#include "include/Dialect/BGV/IR/BGVOps.cpp.inc"

namespace mlir {
namespace heir {
namespace bgv {

//===----------------------------------------------------------------------===//
// BGV dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void BGVDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "include/Dialect/BGV/IR/BGVAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "include/Dialect/BGV/IR/BGVTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/BGV/IR/BGVOps.cpp.inc"
      >();
}

LogicalResult MulOp::verify() {
  auto x = getX().getType();
  auto y = getY().getType();
  if (x.getDim() != y.getDim()) {
    return emitOpError() << "input dimensions do not match";
  }
  auto out = getOutput().getType();
  if (out.getDim() != 1 + x.getDim()) {
    return emitOpError() << "output.dim == x.dim + 1 does not hold";
  }
  return success();
}
LogicalResult Rotate::verify() {
  auto x = getX().getType();
  if (x.getDim() != 2) {
    return emitOpError() << "x.dim == 2 does not hold";
  }
  auto out = getOutput().getType();
  if (out.getDim() != 2) {
    return emitOpError() << "output.dim == 2 does not hold";
  }
  return success();
}

LogicalResult Relinearize::verify() {
  auto x = getX().getType();
  auto out = getOutput().getType();
  if (x.getDim() != getFromBasis().size()) {
    return emitOpError() << "input dimension does not match from_basis";
  }
  if (out.getDim() != getToBasis().size()) {
    return emitOpError() << "output dimension does not match to_basis";
  }
  return success();
}

LogicalResult ModulusSwitch::verify() {
  auto x = getX().getType();
  auto rings = x.getRings().getRings().size();
  auto to = getToLevel();
  auto from = getFromLevel();
  if (to < 0 || to >= from || from >= rings) {
    return emitOpError() << "invalid levels, should be true: 0 <= " << to
                         << " < " << from << " < " << rings;
  }
  if (x.getLevel().has_value() && x.getLevel().value() != from) {
    return emitOpError() << "input level does not match from_level";
  }
  auto outLvl = getOutput().getType().getLevel();
  if (!outLvl.has_value() || outLvl.value() != to) {
    return emitOpError()
           << "output level should be specified and match to_level";
  }
  return success();
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
