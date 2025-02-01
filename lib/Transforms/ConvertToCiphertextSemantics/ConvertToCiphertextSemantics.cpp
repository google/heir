#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"

#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

#define DEBUG_TYPE "linalg-to-tensor-ext"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_CONVERTTOCIPHERTEXTSEMANTICS
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h.inc"

bool isPowerOfTwo(int64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

// This type converter converts types like tensor<NxMxi16> where the dimensions
// represent tensor-semantic data to tensor<ciphertext_count x num_slots x
// i16>, where the last dimension represents the ciphertext or plaintext slot
// count, and the other dimensions are determined by a layout attribute
// indexing.
struct LayoutMaterializationTypeConverter : public AttributeAwareTypeConverter {
 public:
  LayoutMaterializationTypeConverter(int numSlots) : numSlots(numSlots) {}

 private:
  // The number of slots available in each ciphertext.
  int numSlots;
};

struct ConvertToCiphertextSemantics
    : impl::ConvertToCiphertextSemanticsBase<ConvertToCiphertextSemantics> {
  using ConvertToCiphertextSemanticsBase::ConvertToCiphertextSemanticsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    LayoutMaterializationTypeConverter typeConverter;

    // RewritePatternSet patterns(context);

    // (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
