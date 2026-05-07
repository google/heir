#ifndef LIB_TARGET_LATTIGO_LATTIGODEBUGEMITTER_H_
#define LIB_TARGET_LATTIGO_LATTIGODEBUGEMITTER_H_

#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

using ErrorEmitterFn = std::function<LogicalResult(::mlir::Location loc,
                                                   const std::string& message)>;

/// Translates the given operation to Lattigo
::mlir::LogicalResult translateToDebugEmitter(::mlir::Operation* op,
                                              llvm::raw_ostream& os,
                                              const std::string& packageName);

class LattigoDebugEmitter {
 public:
  LattigoDebugEmitter(raw_ostream& os, const std::string& packageName);

  LogicalResult translate(::mlir::Operation& operation);

  void emitPrelude() {
    os << "package " << packageName << "\n";
    os << "import (\n";
    for (const auto& import : imports) {
      os << "  " << import << "\n";
    }
    os << ")\n";
    os << "\n";
  }

 private:
  /// Output stream to emit to.
  raw_indented_ostream os;

  const std::string& packageName;
  std::string prelude;
  std::set<std::string> imports;

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);
  LogicalResult printOperation(::mlir::func::FuncOp op);

  // Emit the default debug helper function signature
  LogicalResult emitDebugHelperSignature(::mlir::func::FuncOp funcOp,
                                         ErrorEmitterFn emitError);

  LogicalResult emitDebugHelperImpl();

  FailureOr<std::string> convertType(::mlir::Type type);
  bool isEmitted;
};

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_LATTIGO_LATTIGODEBUGEMITTER_H_
