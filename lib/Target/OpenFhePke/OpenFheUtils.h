#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEUTILS_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEUTILS_H_

#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

inline constexpr llvm::StringLiteral kDebugAttrMapParam = "debugAttrMap";
inline constexpr llvm::StringLiteral kIsBlockArgVar = "isBlockArgument";
inline constexpr llvm::StringLiteral kCctxtVar = "cc";
inline constexpr llvm::StringLiteral kPrivKeyTVar = "sk";
inline constexpr llvm::StringLiteral kCiphertxtVar = "ct";
inline constexpr llvm::StringLiteral kPlaintxtVar = "ptxt";

enum class OpenfheScheme { BGV, BFV, CKKS };

// OpenFHE's installation process moves headers around in the install directory,
// as well as changing the import paths from the development repository. This
// option controls which type of import should be used on the generated code.
enum class OpenfheImportType {
  // Import paths are relative to the openfhe development repository, i.e.,
  // paths like #include "src/pke/include/openfhe.h". This is primarily useful
  // for development within HEIR, where the openfhe source repository is cloned
  // by bazel and otherwise not installed on the system.
  SOURCE_RELATIVE,

  // Import paths are relative to the openfhe installation process, i.e., paths
  // like #include "openfhe/pke/openfhe.h". This is useful for user-facing code
  // generation, where the openfhe backend is installed by the user or shipped
  // as a shared library dependency of a heir frontend.
  INSTALL_RELATIVE,

  // Import paths are embedded directly in the generated code, i.e., paths like
  // #include "openfhe.h". This is useful for generating code that is intended
  // to be put in openfhe source files, where the openfhe headers are already
  // handled by the build system.
  EMBEDDED,
};

std::string getModulePrelude(OpenfheScheme scheme,
                             OpenfheImportType importType);

std::string getWeightsPrelude();

/// Convert a type to a string, using a const specifier if constant is true.
::mlir::FailureOr<std::string> convertType(::mlir::Type type,
                                           ::mlir::Location loc,
                                           bool constant = false);

/// Find the CryptoContext SSA value in the input operation's parent func
/// arguments.
::mlir::FailureOr<::mlir::Value> getContextualCryptoContext(
    ::mlir::Operation* op);

using ErrorEmitterFn = std::function<LogicalResult(::mlir::Location loc,
                                                   const std::string& message)>;
using TypeEmitterFn =
    std::function<LogicalResult(::mlir::Type type, ::mlir::Location loc)>;

// Emit the shared parts of a function declaration between its header
// declaration and its definition. I.e., stop at the closing parent
// after the argument list, and let the caller decide whether to emit
// a following semicolon or function body.
LogicalResult funcDeclarationHelper(::mlir::func::FuncOp funcOp,
                                    ::mlir::raw_indented_ostream& os,
                                    SelectVariableNames* variableNames,
                                    TypeEmitterFn emitType,
                                    ErrorEmitterFn emitError);

// Emit the default debug helper function signature
LogicalResult emitDebugHelperSignature(::mlir::func::FuncOp funcOp,
                                       ::mlir::raw_indented_ostream& os,
                                       ErrorEmitterFn emitError);

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEUTILS_H_
