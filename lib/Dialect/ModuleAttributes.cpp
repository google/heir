#include "lib/Dialect/ModuleAttributes.h"

#include <algorithm>

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "mlir/include/mlir/IR/Attributes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {

DictionaryAttr getInterfaceAttr(Operation* op, StringRef role) {
  auto metadata = op->getAttrOfType<DictionaryAttr>(kInterfaceAttrName);
  if (!metadata || role.empty()) return metadata;
  auto roles = metadata.getAs<ArrayAttr>(kInterfaceRoles);
  if (roles) {
    for (Attribute value : roles)
      if (value == StringAttr::get(op->getContext(), role)) return metadata;
  }
  return {};
}

bool hasInterfaceRole(Operation* op, StringRef role) {
  return static_cast<bool>(getInterfaceAttr(op, role));
}

void setInterfaceRole(Operation* op, StringRef role, DictionaryAttr metadata) {
  NamedAttrList attributes(getInterfaceAttr(op));
  SmallVector<Attribute> roles;
  if (auto current =
          dyn_cast_or_null<ArrayAttr>(attributes.get(kInterfaceRoles)))
    roles.append(current.begin(), current.end());
  auto value = StringAttr::get(op->getContext(), role);
  if (std::find(roles.begin(), roles.end(), value) == roles.end())
    roles.push_back(value);
  for (NamedAttribute attr : metadata)
    if (attr.getName().getValue() != kInterfaceRoles)
      attributes.set(attr.getName(), attr.getValue());
  attributes.set(kInterfaceRoles, ArrayAttr::get(op->getContext(), roles));
  op->setAttr(kInterfaceAttrName, attributes.getDictionary(op->getContext()));
}

void removeInterfaceRole(Operation* op, StringRef role) {
  if (!hasInterfaceRole(op, role)) return;
  NamedAttrList attributes(getInterfaceAttr(op));
  auto current = cast<ArrayAttr>(attributes.get(kInterfaceRoles));
  SmallVector<Attribute> roles(current.begin(), current.end());
  auto value = StringAttr::get(op->getContext(), role);
  roles.erase(std::remove(roles.begin(), roles.end(), value), roles.end());
  if (roles.empty()) {
    op->removeAttr(kInterfaceAttrName);
    return;
  }
  attributes.set(kInterfaceRoles, ArrayAttr::get(op->getContext(), roles));
  op->setAttr(kInterfaceAttrName, attributes.getDictionary(op->getContext()));
}

void setInterfaceField(Operation* op, StringRef name, Attribute value) {
  NamedAttrList attributes(getInterfaceAttr(op));
  attributes.set(name, value);
  op->setAttr(kInterfaceAttrName, attributes.getDictionary(op->getContext()));
}

/*===----------------------------------------------------------------------===*/
// Module Attributes for Scheme
/*===----------------------------------------------------------------------===*/

bool moduleIsBGV(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kBGVSchemeAttrName) != nullptr;
}

bool moduleIsBFV(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kBFVSchemeAttrName) != nullptr;
}

bool moduleIsBGVOrBFV(Operation* moduleOp) {
  return moduleIsBGV(moduleOp) || moduleIsBFV(moduleOp);
}

bool moduleIsCKKS(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kCKKSSchemeAttrName) !=
         nullptr;
}

bool moduleIsCGGI(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kCGGISchemeAttrName) !=
         nullptr;
}

Attribute getSchemeParamAttr(Operation* op) {
  SmallVector<StringLiteral> schemeAttrNames = {
      bgv::BGVDialect::kSchemeParamAttrName,
      ckks::CKKSDialect::kSchemeParamAttrName,
  };

  Operation* moduleOp = op;
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

void moduleClearScheme(Operation* moduleOp) {
  moduleOp->removeAttr(kBGVSchemeAttrName);
  moduleOp->removeAttr(kBFVSchemeAttrName);
  moduleOp->removeAttr(kCKKSSchemeAttrName);
  moduleOp->removeAttr(kCGGISchemeAttrName);
}

void moduleSetBGV(Operation* moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kBGVSchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetBFV(Operation* moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kBFVSchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetCKKS(Operation* moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kCKKSSchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetCGGI(Operation* moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kCGGISchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

/*===----------------------------------------------------------------------===*/
// Module Attributes for Backend
/*===----------------------------------------------------------------------===*/

bool moduleIsOpenfhe(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kOpenfheBackendAttrName) !=
         nullptr;
}

bool moduleIsLattigo(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kLattigoBackendAttrName) !=
         nullptr;
}

bool moduleIsCheddar(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kCheddarBackendAttrName) !=
         nullptr;
}

void moduleClearBackend(Operation* moduleOp) {
  moduleOp->removeAttr(kOpenfheBackendAttrName);
  moduleOp->removeAttr(kLattigoBackendAttrName);
  moduleOp->removeAttr(kCheddarBackendAttrName);
}

void moduleSetOpenfhe(Operation* moduleOp) {
  moduleClearBackend(moduleOp);
  moduleOp->setAttr(kOpenfheBackendAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetLattigo(Operation* moduleOp) {
  moduleClearBackend(moduleOp);
  moduleOp->setAttr(kLattigoBackendAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetCheddar(Operation* moduleOp) {
  moduleClearBackend(moduleOp);
  moduleOp->setAttr(kCheddarBackendAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

}  // namespace heir
}  // namespace mlir
