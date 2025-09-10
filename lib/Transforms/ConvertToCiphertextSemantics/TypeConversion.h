#ifndef LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_TYPECONVERSION_H_
#define LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_TYPECONVERSION_H_

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project

namespace mlir {
namespace heir {

// This lib contains the parts of the layout materialization type converter
// that are reused outside of the main convert-to-ciphertext-semantics pass. In
// particular, by the pattern to lower AssignLayoutOp, which is also reused in
// add-lwe-client-interface.

Type materializeScalarLayout(Type type, tensor_ext::LayoutAttr attr,
                             int ciphertextSize);

Type materializeLayout(RankedTensorType type, tensor_ext::LayoutAttr attr,
                       int ciphertextSize);

Type materializeNewLayout(Type dataType, tensor_ext::NewLayoutAttr attr,
                          int ciphertextSize);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_TYPECONVERSION_H_
