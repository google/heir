#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEPKEDEBUGEMITTER_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEPKEDEBUGEMITTER_H_

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>


#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

/// Translates the given operation to OpenFhePke.
::mlir::LogicalResult translateToOpenFhePkeDebugEmitter(::mlir::Operation* op,
                                            llvm::raw_ostream& os,
                                            OpenfheImportType importType,
                                            const std::string& debugImportPath);


class OpenFhePkeDebugEmitter {
 public:
  OpenFhePkeDebugEmitter(raw_ostream& os,
                    OpenfheImportType importType, 
                    const std::string& debugImportPath);

  LogicalResult translate(::mlir::Operation& operation);

 private:
  OpenfheImportType importType_;

  /// Output stream to emit to.
  raw_indented_ostream os;

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);
  LogicalResult printOperation(::mlir::func::FuncOp op);

  LogicalResult emitDebugHelperImpl();

  /// Include path for debug imports
  std::string debugImportPath;

  bool isEmitted;

};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEPKEDEBUGEMITTER_H_