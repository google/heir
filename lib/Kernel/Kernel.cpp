#include "lib/Kernel/Kernel.h"

#include <ostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "lib/Kernel/KernelName.h"
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"     // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"  // from @llvm-project

#define DEBUG_TYPE "kernel"

namespace mlir {
namespace heir {

namespace {
static std::unordered_map<KernelName, std::vector<std::string>>
    correspondingOp = {
        {KernelName::MatvecNaive, {"linalg.matvec"}},
        {KernelName::MatvecDiagonal,
         {"linalg.matvec", "linalg.conv_2d_nchw_fchw"}},
        {KernelName::VecmatDiagonal, {"linalg.vecmat"}},
        {KernelName::MatmulDiagonal, {"linalg.matmul"}},
        {KernelName::MatmulDiagonal, {"linalg.conv2d"}},
        {KernelName::MatmulBicyclic, {"linalg.matmul"}},
};

std::set<std::string> requiredNontrivial = {"linalg"};
}  // namespace

bool isSupportedKernel(Operation* op, KernelName name) {
  std::string dialect = std::string(op->getDialect()->getNamespace());
  if (name == KernelName::Trivial) {
    return requiredNontrivial.count(dialect) == 0;
  }

  auto it = correspondingOp.find(name);
  if (it == correspondingOp.end()) {
    LLVM_DEBUG(llvm::dbgs() << "Kernel name " << kernelNameAsStr(name)
                            << "not found in correspondingOp legality map\n");
    return false;
  }

  std::string actual;
  llvm::raw_string_ostream ss(actual);
  ss << op->getName().getStringRef();

  auto opForKernelIt = llvm::find(it->second, actual);
  if (opForKernelIt != it->second.end()) {
    return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "Kernel " << kernelNameAsStr(name)
                          << " is not legal for op " << actual
                          << ", expected one of ops: "
                          << llvm::join(it->second, ", ") << "\n");
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
    case KernelName::MatmulDiagonal:
      return "MatmulDiagonal";
    case KernelName::VecmatDiagonal:
      return "VecmatDiagonal";
    case KernelName::MatmulBicyclic:
      return "MatmulBicyclic";
    default:
      return "Unknown";
  }
}

std::ostream& operator<<(std::ostream& os, const KernelName& k) {
  return os << "\"" << kernelNameAsStr(k) << "\"";
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const KernelName& k) {
  return os << "\"" << kernelNameAsStr(k) << "\"";
}

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const KernelName& k) {
  return diag << kernelNameAsStr(k);
}

}  // namespace heir
}  // namespace mlir
