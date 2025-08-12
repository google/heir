#ifndef LIB_KERNEL_KERNEL_H_
#define LIB_KERNEL_KERNEL_H_

#include <string>

#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

enum class KernelName {
  // A trivial kernel is one for which there is only a single known option
  // (e.g., an elementwise addition).
  Trivial,
  MatvecNaive,
  MatvecDiagonal,
};

bool isSupportedKernel(Operation* op, KernelName name);

inline raw_ostream& operator<<(raw_ostream& os,
                               const heir::KernelName& kernelName) {
  switch (kernelName) {
    case heir::KernelName::Trivial:
      os << "Trivial";
      break;
    case heir::KernelName::MatvecNaive:
      os << "MatvecNaive";
      break;
    case heir::KernelName::MatvecDiagonal:
      os << "MatvecDiagonal";
      break;
    default:
      os << "Unknown";
  }
  return os;
}

}  // namespace heir

template <>
struct FieldParser<heir::KernelName> {
  static FailureOr<heir::KernelName> parse(AsmParser& parser) {
    std::string kernelName;
    if (parser.parseString(&kernelName)) return failure();

    if (kernelName == "Trivial") return heir::KernelName::Trivial;
    if (kernelName == "MatvecNaive") return heir::KernelName::MatvecNaive;
    if (kernelName == "MatvecDiagonal") return heir::KernelName::MatvecDiagonal;

    return failure();
  }
};

}  // namespace mlir
#endif  // LIB_KERNEL_KERNEL_H_
