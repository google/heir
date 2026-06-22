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
    case KernelName::RotomAdd:
      return "RotomAdd";
    case KernelName::RotomMul:
      return "RotomMul";
    case KernelName::Dot:
      return "Dot";
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
