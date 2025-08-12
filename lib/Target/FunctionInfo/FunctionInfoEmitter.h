#ifndef LIB_TARGET_FUNCTIONINFO_FUNCTIONINFOEMITTER_H_
#define LIB_TARGET_FUNCTIONINFO_FUNCTIONINFOEMITTER_H_

#include <string>

#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace functioninfo {

void registerToFunctionInfoTranslation();

LogicalResult translateToFunctionInfo(Operation* op, llvm::raw_ostream& os);

/// FunctionInfoEmitter is a simple helper (created for use with the python
/// frontend) that reads the first func.func in the module and emits the
/// following information: function_name x, y, z, ... (argument names - though
/// currently, these will just be arg_0, etc) 2, 3, ... (indices of arguments
/// which are "secret" in the HEIR sense)
//
// TODO (#1162): Investigate how we can preserve the custom ssa names from the
// mlir str
class FunctionInfoEmitter {
 public:
  FunctionInfoEmitter(llvm::raw_ostream& os);
  LogicalResult translate(Operation& operation);

 private:
  raw_indented_ostream os;

  LogicalResult printOperation(ModuleOp moduleOp);
};

}  // namespace functioninfo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_FUNCTIONINFO_FUNCTIONINFOEMITTER_H_
