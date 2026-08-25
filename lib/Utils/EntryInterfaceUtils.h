#ifndef LIB_UTILS_ENTRYINTERFACEUTILS_H_
#define LIB_UTILS_ENTRYINTERFACEUTILS_H_

#include <optional>
#include <string>
#include <utility>

#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {

// The functions making up one entry point's client/server interface, keyed by
// the role attributes in ModuleAttributes.h. Reading these is how a backend
// builds a public API without inspecting lowered signatures or symbol names.
struct EntryFunctions {
  std::string entryName;
  func::FuncOp contract;    // heir.entry_func; null below the secret level
  func::FuncOp setup;       // client.setup_func
  func::FuncOp keygen;      // client.keygen_func; null when setup does keygen
  func::FuncOp preprocess;  // server.preprocessing_func
  func::FuncOp evaluate;    // server.evaluate_func
  // client.enc_func / indexed client.pack_func, by entry-argument index.
  SmallVector<std::pair<unsigned, func::FuncOp>> inputHelpers;
  // client.dec_func, by entry-result index.
  SmallVector<std::pair<unsigned, func::FuncOp>> outputHelpers;
  // client.enc_zero_func, by index; each pairs with the entry argument
  // carrying client.enc_zero_arg at the same index.
  SmallVector<std::pair<unsigned, func::FuncOp>> zeroHelpers;
};

DictionaryAttr getRoleAttr(func::FuncOp function, StringRef name);

std::optional<StringRef> getRoleEntry(func::FuncOp function, StringRef name);

FailureOr<unsigned> getRoleIndex(func::FuncOp function, StringRef name);

// heir.entry_input_types / heir.entry_result_types.
ArrayAttr getLogicalTypes(func::FuncOp function, StringRef name);

func::FuncOp findIndexedHelper(
    ArrayRef<std::pair<unsigned, func::FuncOp>> helpers, unsigned index);

LogicalResult validateIndexedHelpers(
    ArrayRef<std::pair<unsigned, func::FuncOp>> helpers, unsigned count,
    StringRef kind, Operation* diagnostic);

// Collects the roles for `requestedEntry`, or for the module's only entry when
// it is empty. Only the evaluate role is required, so a backend can emit a
// partial interface when the others are absent.
FailureOr<EntryFunctions> findEntryFunctions(ModuleOp module,
                                             StringRef requestedEntry);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_ENTRYINTERFACEUTILS_H_
