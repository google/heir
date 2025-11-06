#ifndef LIB_KERNEL_KERNEL_H_
#define LIB_KERNEL_KERNEL_H_

#include <string>

#include "lib/Kernel/KernelName.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

bool isSupportedKernel(Operation* op, KernelName name);

std::string kernelNameAsStr(const heir::KernelName& kernelName);

}  // namespace heir

template <>
struct FieldParser<heir::KernelName> {
  static FailureOr<heir::KernelName> parse(AsmParser& parser) {
    std::string kernelName;
    if (parser.parseString(&kernelName)) return failure();

    if (kernelName == "Trivial") return heir::KernelName::Trivial;
    if (kernelName == "MatvecNaive") return heir::KernelName::MatvecNaive;
    if (kernelName == "MatvecDiagonal") return heir::KernelName::MatvecDiagonal;
    if (kernelName == "VecmatDiagonal") return heir::KernelName::VecmatDiagonal;
    if (kernelName == "MatmulDiagonal") return heir::KernelName::MatmulDiagonal;
    if (kernelName == "MatmulBicyclic") return heir::KernelName::MatmulBicyclic;

    return failure();
  }
};

}  // namespace mlir
#endif  // LIB_KERNEL_KERNEL_H_
