#ifndef LIB_DIALECT_MODULEATTRIBUTES_H_
#define LIB_DIALECT_MODULEATTRIBUTES_H_

#include "llvm/include/llvm/ADT/StringRef.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project

namespace mlir {
namespace heir {

/*===----------------------------------------------------------------------===*/
// Module Attributes for Scheme
/*===----------------------------------------------------------------------===*/

// These attributes are intended to be set early on in the compilation pipeline,
// and checked by individual passes that need to know the target scheme. This
// avoids threading CLI flags through pass and sub-pipeline options.
constexpr const static ::llvm::StringLiteral kBGVSchemeAttrName = "scheme.bgv";
constexpr const static ::llvm::StringLiteral kBFVSchemeAttrName = "scheme.bfv";
constexpr const static ::llvm::StringLiteral kCKKSSchemeAttrName =
    "scheme.ckks";
constexpr const static ::llvm::StringLiteral kCGGISchemeAttrName =
    "scheme.cggi";

constexpr const static ::llvm::StringLiteral kRequestedSlotCountAttrName =
    "scheme.requested_slot_count";
constexpr const static ::llvm::StringLiteral kActualSlotCountAttrName =
    "scheme.actual_slot_count";

bool moduleIsBGV(Operation* moduleOp);
bool moduleIsBFV(Operation* moduleOp);
bool moduleIsBGVOrBFV(Operation* moduleOp);
bool moduleIsCKKS(Operation* moduleOp);
bool moduleIsCGGI(Operation* moduleOp);

// Fetch the scheme parameter attribute from the parent module op. This
// parameter is only set on the module after a parameter selection pass runs.
Attribute getSchemeParamAttr(Operation* op);

void moduleClearScheme(Operation* moduleOp);

void moduleSetBGV(Operation* moduleOp);
void moduleSetBFV(Operation* moduleOp);
void moduleSetCKKS(Operation* moduleOp);
void moduleSetCGGI(Operation* moduleOp);

/*===----------------------------------------------------------------------===*/
// Module Attributes for Backend
/*===----------------------------------------------------------------------===*/

// Similar to the scheme attributes, these indicate the target backend for
// passes to branch behavior on.
constexpr const static ::llvm::StringLiteral kOpenfheBackendAttrName =
    "backend.openfhe";
constexpr const static ::llvm::StringLiteral kLattigoBackendAttrName =
    "backend.lattigo";
constexpr const static ::llvm::StringLiteral kCheddarBackendAttrName =
    "backend.cheddar";

bool moduleIsOpenfhe(Operation* moduleOp);
bool moduleIsLattigo(Operation* moduleOp);
bool moduleIsCheddar(Operation* moduleOp);

void moduleClearBackend(Operation* moduleOp);

void moduleSetOpenfhe(Operation* moduleOp);
void moduleSetLattigo(Operation* moduleOp);
void moduleSetCheddar(Operation* moduleOp);

// A function's client/server interface metadata lives in one dictionary:
// heir.interface = {func_name = "foo", roles = ["entry", "server.evaluate"],
//                   input_types = [...], result_types = [...]}
// Indexed helpers add `index`; preprocessing adds `entry_arg_indices`.
constexpr const static ::llvm::StringLiteral kInterfaceAttrName =
    "heir.interface";
constexpr const static ::llvm::StringLiteral kInterfaceRoles = "roles";

// An absent role returns a null dictionary. An empty role selects all metadata.
DictionaryAttr getInterfaceAttr(Operation* op, StringRef role = {});
bool hasInterfaceRole(Operation* op, StringRef role);
void setInterfaceRole(Operation* op, StringRef role, DictionaryAttr metadata);
void removeInterfaceRole(Operation* op, StringRef role);
void setInterfaceField(Operation* op, StringRef name, Attribute value);

constexpr const static ::llvm::StringLiteral kClientEncRole = "client.encrypt";
constexpr const static ::llvm::StringLiteral kClientDecRole = "client.decrypt";
constexpr const static ::llvm::StringLiteral kClientPackRole = "client.pack";
// The zero-encryption helper and its entry argument share func_name and index.
constexpr const static ::llvm::StringLiteral kClientEncZeroRole =
    "client.encrypt_zero";
constexpr const static ::llvm::StringLiteral kClientEncZeroArgAttrName =
    "client.enc_zero_arg";

constexpr const static ::llvm::StringLiteral kClientSetupRole = "client.setup";
constexpr const static ::llvm::StringLiteral kClientKeygenRole =
    "client.keygen";

// Roles share the logical entry identity in `func_name`.
constexpr const static ::llvm::StringLiteral kEntryRole = "entry";
constexpr const static ::llvm::StringLiteral kServerPreprocessingRole =
    "server.preprocessing";
// For the server.preprocessing role: the entry argument each
// preprocessing parameter is forwarded from, in parameter order (-1 if none).
constexpr const static ::llvm::StringLiteral kServerPreprocessingEntryArgs =
    "entry_arg_indices";
constexpr const static ::llvm::StringLiteral kServerEvaluateRole =
    "server.evaluate";
constexpr const static ::llvm::StringLiteral kServerSetupRole = "server.setup";

// Arrays of TypeAttr preserving the original cleartext entry signature.
constexpr const static ::llvm::StringLiteral kEntryInputTypes = "input_types";
constexpr const static ::llvm::StringLiteral kEntryResultTypes = "result_types";

// Marks the ciphertext workload after plaintext preprocessing is split out.
constexpr const static ::llvm::StringLiteral kClientPreprocessedRole =
    "client.preprocessed";

inline bool isClientHelper(Operation* op) {
  return hasInterfaceRole(op, kClientEncRole) ||
         hasInterfaceRole(op, kClientDecRole) ||
         hasInterfaceRole(op, kClientPackRole) ||
         hasInterfaceRole(op, kClientSetupRole) ||
         hasInterfaceRole(op, kClientKeygenRole) ||
         hasInterfaceRole(op, kServerSetupRole) ||
         hasInterfaceRole(op, kServerPreprocessingRole) ||
         hasInterfaceRole(op, kClientPreprocessedRole) ||
         hasInterfaceRole(op, kClientEncZeroRole);
}

// The name of the function this client helper is made for.
constexpr const static ::llvm::StringLiteral kClientHelperFuncName =
    "func_name";
// The argument or operand index the client helper function is for.
constexpr const static ::llvm::StringLiteral kClientHelperIndex = "index";

inline bool isPreprocessingHelper(Operation* op) {
  return hasInterfaceRole(op, kClientPackRole) ||
         hasInterfaceRole(op, kServerPreprocessingRole);
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MODULEATTRIBUTES_H_
