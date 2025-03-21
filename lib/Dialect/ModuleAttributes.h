#ifndef LIB_DIALECT_MODULEATTRIBUTES_H_
#define LIB_DIALECT_MODULEATTRIBUTES_H_

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"   // from @llvm-project

namespace mlir {
namespace heir {

/*===----------------------------------------------------------------------===*/
// Module Attributes for Scheme
/*===----------------------------------------------------------------------===*/

constexpr const static ::llvm::StringLiteral kBGVSchemeAttrName = "scheme.bgv";
constexpr const static ::llvm::StringLiteral kBFVSchemeAttrName = "scheme.bfv";
constexpr const static ::llvm::StringLiteral kCKKSSchemeAttrName =
    "scheme.ckks";
constexpr const static ::llvm::StringLiteral kCGGISchemeAttrName =
    "scheme.cggi";

bool moduleIsBGV(Operation *moduleOp);
bool moduleIsBFV(Operation *moduleOp);
bool moduleIsBGVOrBFV(Operation *moduleOp);
bool moduleIsCKKS(Operation *moduleOp);
bool moduleIsCGGI(Operation *moduleOp);

void moduleClearScheme(Operation *moduleOp);

void moduleSetBGV(Operation *moduleOp);
void moduleSetBFV(Operation *moduleOp);
void moduleSetCKKS(Operation *moduleOp);
void moduleSetCGGI(Operation *moduleOp);

/*===----------------------------------------------------------------------===*/
// Module Attributes for Backend
/*===----------------------------------------------------------------------===*/

constexpr const static ::llvm::StringLiteral kOpenfheBackendAttrName =
    "backend.openfhe";
constexpr const static ::llvm::StringLiteral kLattigoBackendAttrName =
    "backend.lattigo";

bool moduleIsOpenfhe(Operation *moduleOp);
bool moduleIsLattigo(Operation *moduleOp);

void moduleClearBackend(Operation *moduleOp);

void moduleSetOpenfhe(Operation *moduleOp);
void moduleSetLattigo(Operation *moduleOp);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MODULEATTRIBUTES_H_
