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
  auto x = this->getX().getType();
  auto y = this->getY().getType();
  if (x.getDim() != y.getDim()) {
    return this->emitOpError() << "input dimensions do not match";
  }
  auto out = this->getOutput().getType();
  if (out.getDim() != 1 + x.getDim()) {
    return this->emitOpError() << "output.dim == x.dim + 1 does not hold";
  }
  return success();
}

LogicalResult Relinearize::verify() {
  auto x = this->getX().getType();
  auto out = this->getOutput().getType();
  if (x.getDim() != this->getFromBasis().size()) {
    return this->emitOpError() << "input dimension does not match from_basis";
  }
  if (out.getDim() != this->getToBasis().size()) {
    return this->emitOpError() << "output dimension does not match to_basis";
  }
  return success();
}

LogicalResult ModulusSwitch::verify() {
  auto x = this->getX().getType();
  auto rings = x.getRings().getRings().size();
  auto to = this->getToLevel();
  auto from = this->getFromLevel();
  if (to < 0 || to >= from || from >= rings) {
    return this->emitOpError() << "invalid levels, should be true: 0 <= " << to
                               << " < " << from << " < " << rings;
  }
  if (x.getLevel().has_value() && x.getLevel().value() != from) {
    return this->emitOpError() << "input level does not match from_level";
  }
  auto outLvl = this->getOutput().getType().getLevel();
  if (!outLvl.has_value() || outLvl.value() != to) {
    return this->emitOpError()
           << "output level should be specified and match to_level";
  }
  return success();
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
