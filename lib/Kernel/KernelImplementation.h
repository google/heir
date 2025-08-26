#ifndef LIB_KERNEL_KERNELIMPLEMENTATION_H_
#define LIB_KERNEL_KERNELIMPLEMENTATION_H_

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "lib/Kernel/KernelName.h"
#include "lib/Utils/ArithmeticDag.h"
#include "mlir/include/mlir/IR/Value.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

// A variant type that holds a "ciphertext semantic tensor," which can be
// thought of as the packed representation of a ciphertext or plaintext
// (encrypted or not).
//
// More variants must be added to support higher-dimensional input/output
// tensors.
using CiphertextSemanticTensor =
    std::variant<std::vector<double>, std::vector<std::vector<double>>>;

// A struct that contains a tensor's shape and either its MLIR value or its
// literal value. The literal value can be used for testing.
struct ValueOrLiteral {
  std::vector<int64_t> shape;
  std::variant<::mlir::Value, CiphertextSemanticTensor> value;
};

using KernelImplementation = ArithmeticDagNode<ValueOrLiteral>;

// Returns an arithmetic DAG that implements a kernel with the given input
// tensor types.
std::shared_ptr<KernelImplementation> implementKernel(KernelName kernelName,
                                                      ValueOrLiteral matrix,
                                                      ValueOrLiteral vector);

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_KERNELIMPLEMENTATION_H_
