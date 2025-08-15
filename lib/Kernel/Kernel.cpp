#include "lib/Kernel/Kernel.h"

#include <set>
#include <string>
#include <unordered_map>

#include "lib/Kernel/KernelName.h"
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"  // from @llvm-project

#define DEBUG_TYPE "kernel"

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
    LLVM_DEBUG(llvm::dbgs() << "Kernel name " << kernelNameAsStr(name)
                            << "not found in correspondingOp legality map\n");
    return false;
  }

  std::string actual;
  llvm::raw_string_ostream ss(actual);
  ss << op->getName().getStringRef();

  std::string resolvedOpName = correspondingOp.at(name);
  if (resolvedOpName == actual) {
    return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "Kernel " << kernelNameAsStr(name)
                          << " is not legal for op " << actual << ", requires "
                          << resolvedOpName << "\n");
  return false;
}

std::string kernelNameAsStr(const KernelName& kernelName) {
  switch (kernelName) {
    case KernelName::Trivial:
      return "Trivial";
    case KernelName::MatvecNaive:
      return "MatvecNaive";
    case KernelName::MatvecDiagonal:
      return "MatvecDiagonal";
    default:
      return "Unknown";
  }
}

}  // namespace heir

std::ostream& operator<<(std::ostream& os, const heir::KernelName& k) {
  return os << heir::kernelNameAsStr(k);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const heir::KernelName& k) {
  return os << heir::kernelNameAsStr(k);
}

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag,
                             const heir::KernelName& k) {
  return diag << heir::kernelNameAsStr(k);
}

}  // namespace mlir
