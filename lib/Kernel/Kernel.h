#ifndef LIB_KERNEL_KERNEL_H_
#define LIB_KERNEL_KERNEL_H_

#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project

namespace mlir {
namespace heir {

enum class KernelName {
  MatvecNaive,
  MatvecDiagonal,
};

bool isSupportedKernel(Operation *op, KernelName name);

inline raw_ostream &operator<<(raw_ostream &os,
                               const heir::KernelName &kernelName) {
  switch (kernelName) {
    case heir::KernelName::MatvecNaive:
      os << "MatvecNaive";
    case heir::KernelName::MatvecDiagonal:
      os << "MatvecDiagonal";
    default:
      os << "Unknown";
  }
  return os;
}

}  // namespace heir

template <>
struct FieldParser<heir::KernelName> {
  static FailureOr<heir::KernelName> parse(AsmParser &parser) {
    std::string kernelName;
    if (parser.parseString(&kernelName)) return failure();

    if (kernelName == "MatvecNaive") return heir::KernelName::MatvecNaive;
    if (kernelName == "MatvecDiagonal") return heir::KernelName::MatvecDiagonal;

    return failure();
  }
};

}  // namespace mlir
#endif  // LIB_KERNEL_KERNEL_H_
