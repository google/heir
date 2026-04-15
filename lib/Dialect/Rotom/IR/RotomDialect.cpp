#include "lib/Dialect/Rotom/IR/RotomDialect.h"

// IYWU pragma: begin_keep
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.cpp.inc"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
// IYWU pragma: end_keep

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Rotom/IR/RotomAttributes.cpp.inc"

namespace mlir {
namespace heir {
namespace rotom {

void RotomDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/Rotom/IR/RotomAttributes.cpp.inc"
      >();
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
