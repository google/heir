#ifndef LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_TYPECONVERSION_H_
#define LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_TYPECONVERSION_H_

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project

namespace mlir {
namespace heir {

// This lib contains the parts of the layout materialization type converter
// that are reused outside of the main convert-to-ciphertext-semantics pass. In
// particular, by the pattern to lower AssignLayoutOp, which is also reused in
// add-client-interface.

Type materializeLayout(Type dataType, tensor_ext::LayoutAttr attr,
                       int ciphertextSize);

Type materializeScalarLayout(Type type, tensor_ext::LayoutAttr attr,
                             int ciphertextSize);

// Computes the ciphertext-semantic type for a permutation layout given as a
// <N x 4 x i64> DenseIntElementsAttr of (src_ct, src_slot, dst_ct, dst_slot)
// tuples. The number of ciphertexts is derived from the max dst_ct value.
Type materializePermutationLayout(Type elementType,
                                  DenseIntElementsAttr permutation,
                                  int ciphertextSize);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_TYPECONVERSION_H_
