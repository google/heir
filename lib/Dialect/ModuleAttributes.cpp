#include "lib/Dialect/ModuleAttributes.h"

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project

namespace mlir {
namespace heir {

/*===----------------------------------------------------------------------===*/
// Module Attributes for Scheme
/*===----------------------------------------------------------------------===*/

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

Attribute getSchemeParamAttr(Operation *op) {
  SmallVector<StringLiteral> schemeAttrNames = {
      bgv::BGVDialect::kSchemeParamAttrName,
      ckks::CKKSDialect::kSchemeParamAttrName,
  };

  Operation *moduleOp = op;
  if (!isa<ModuleOp>(op)) {
    moduleOp = op->getParentOfType<ModuleOp>();
  }

  for (auto schemeAttrName : schemeAttrNames) {
    if (auto schemeAttr = moduleOp->getAttr(schemeAttrName)) {
      return schemeAttr;
    }
  }

  return UnitAttr::get(op->getContext());
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

/*===----------------------------------------------------------------------===*/
// Module Attributes for Backend
/*===----------------------------------------------------------------------===*/

bool moduleIsOpenfhe(Operation *moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kOpenfheBackendAttrName) !=
         nullptr;
}

bool moduleIsLattigo(Operation *moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kLattigoBackendAttrName) !=
         nullptr;
}

void moduleClearBackend(Operation *moduleOp) {
  moduleOp->removeAttr(kOpenfheBackendAttrName);
  moduleOp->removeAttr(kLattigoBackendAttrName);
}

void moduleSetOpenfhe(Operation *moduleOp) {
  moduleClearBackend(moduleOp);
  moduleOp->setAttr(kOpenfheBackendAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetLattigo(Operation *moduleOp) {
  moduleClearBackend(moduleOp);
  moduleOp->setAttr(kLattigoBackendAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

}  // namespace heir
}  // namespace mlir
