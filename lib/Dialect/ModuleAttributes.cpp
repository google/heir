#include "lib/Dialect/ModuleAttributes.h"

#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project

namespace mlir {
namespace heir {

bool moduleIsBGV(Operation *moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kBGVSchemeAttrName) != nullptr;
}

bool moduleIsCKKS(Operation *moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kCKKSSchemeAttrName) !=
         nullptr;
}

bool moduleIsCGGI(Operation *moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kCGGISchemeAttrName) !=
         nullptr;
}

}  // namespace heir
}  // namespace mlir
