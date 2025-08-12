#include "lib/Kernel/Kernel.h"

#include <set>
#include <string>
#include <unordered_map>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"  // from @llvm-project

namespace mlir {
namespace heir {

namespace {
static std::unordered_map<KernelName, std::string> correspondingOp = {
    {KernelName::MatvecNaive, "linalg.matvec"},
    {KernelName::MatvecDiagonal, "linalg.matvec"},
};

std::set<std::string> requiredNontrivial = {"linalg"};
}  // namespace

bool isSupportedKernel(Operation* op, KernelName name) {
  std::string dialect = std::string(op->getDialect()->getNamespace());
  if (name == KernelName::Trivial) {
    return requiredNontrivial.count(dialect) == 0;
  }

  if (correspondingOp.find(name) == correspondingOp.end()) {
    return false;
  }

  std::string actual;
  llvm::raw_string_ostream ss(actual);
  ss << op->getDialect()->getNamespace() << "." << op->getName().getStringRef();
  return correspondingOp.at(name) == actual;
}

}  // namespace heir
}  // namespace mlir
