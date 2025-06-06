#include "Kernel.h"

#include <string>
#include <unordered_map>

#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project

namespace mlir {
namespace heir {

namespace {
static std::unordered_map<KernelName, std::string> correspondingOp = {
    {KernelName::MatvecNaive, "linalg.matvec"},
    {KernelName::MatvecDiagonal, "linalg.matvec"},
};
}  // namespace

bool isSupportedKernel(Operation *op, KernelName name) {
  std::string actual;
  llvm::raw_string_ostream ss(actual);
  ss << op->getDialect()->getNamespace() << "." << op->getName().getStringRef();
  return correspondingOp.at(name) == actual;
}

}  // namespace heir
}  // namespace mlir
