#include "lib/Dialect/ModuleAttributes.h"

#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project

namespace mlir {
namespace heir {

bool moduleIsBGV(Operation *moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kBGVSchemeAttrName) != nullptr;
}

bool moduleIsBFV(Operation *moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kBFVSchemeAttrName) != nullptr;
}

bool moduleIsBGVOrBFV(Operation *moduleOp) {
  return moduleIsBGV(moduleOp) || moduleIsBFV(moduleOp);
}

bool moduleIsCKKS(Operation *moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kCKKSSchemeAttrName) !=
         nullptr;
}

bool moduleIsCGGI(Operation *moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kCGGISchemeAttrName) !=
         nullptr;
}

void moduleClearScheme(Operation *moduleOp) {
  moduleOp->removeAttr(kBGVSchemeAttrName);
  moduleOp->removeAttr(kBFVSchemeAttrName);
  moduleOp->removeAttr(kCKKSSchemeAttrName);
  moduleOp->removeAttr(kCGGISchemeAttrName);
}

void moduleSetBGV(Operation *moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kBGVSchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetBFV(Operation *moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kBFVSchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetCKKS(Operation *moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kCKKSSchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetCGGI(Operation *moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kCGGISchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

}  // namespace heir
}  // namespace mlir
