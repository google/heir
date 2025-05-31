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
constexpr const static ::llvm::StringLiteral kPlaintextSchemeAttrName =
    "plaintext.log_default_scale";

bool moduleIsBGV(Operation *moduleOp);
bool moduleIsBFV(Operation *moduleOp);
bool moduleIsBGVOrBFV(Operation *moduleOp);
bool moduleIsCKKS(Operation *moduleOp);
bool moduleIsCGGI(Operation *moduleOp);

// Fetch the scheme parameter attribute from the parent module op.
Attribute getSchemeParamAttr(Operation *op);

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

// Func attributes for client helpers
//
// This corresponds to a named attribute client.enc_func whose
// value is a dictionary {func_name = "foo", index = 2 : i64}
//
// This means that the function with this attribute is an encryption
// helper for the function "foo" and the argument at index 2.

constexpr const static ::llvm::StringLiteral kClientEncFuncAttrName =
    "client.enc_func";
constexpr const static ::llvm::StringLiteral kClientDecFuncAttrName =
    "client.dec_func";

inline bool isClientHelper(Operation *op) {
  return op->hasAttr(kClientDecFuncAttrName) ||
         op->hasAttr(kClientDecFuncAttrName);
}

// The name of the function this client helper is made for.
constexpr const static ::llvm::StringLiteral kClientHelperFuncName =
    "func_name";
// The argument or operand index the client helper function is for.
constexpr const static ::llvm::StringLiteral kClientHelperIndex = "index";

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MODULEATTRIBUTES_H_
