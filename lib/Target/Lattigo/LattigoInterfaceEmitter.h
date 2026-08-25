#ifndef LIB_TARGET_LATTIGO_LATTIGOINTERFACEEMITTER_H_
#define LIB_TARGET_LATTIGO_LATTIGOINTERFACEEMITTER_H_

#include <string>
#include <vector>

#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

// Emits a Go facade over the generated Lattigo ABI: a context struct holding
// what __configure returns, and Setup/Preprocess/Encrypt/Evaluate/Decrypt
// wrapping the generated helpers. The facade's shape does not change with
// bootstrapping, the number of client-supplied encrypted zeros or the
// preprocessing storage arity, so a harness can be written once against it.
//
// A module missing a role -- one entered at ciphertext semantics has no client
// encryption helpers -- yields no corresponding method.
::mlir::LogicalResult translateToLattigoInterface(
    ::mlir::Operation* op, llvm::raw_ostream& os,
    const std::string& packageName,
    const std::vector<std::string>& extraImports = {},
    const std::string& interfacePrefix = "");

void registerToLattigoInterfaceTranslation();

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_LATTIGO_LATTIGOINTERFACEEMITTER_H_
