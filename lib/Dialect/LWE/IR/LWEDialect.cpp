#include "lib/Dialect/LWE/IR/LWEDialect.h"

#include <cassert>

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"

// IWYU pragma: begin_keep
#include "llvm/include/llvm/ADT/STLExtras.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
// IWYU pragma: end_keep

// Generated definitions
#include "lib/Dialect/LWE/IR/LWEDialect.cpp.inc"
#include "lib/Dialect/LWE/IR/LWEEnums.cpp.inc"
#include "mlir/include/mlir/IR/Operation.h"      // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/LWE/IR/LWEAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/LWE/IR/LWETypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/LWE/IR/LWEOps.cpp.inc"

namespace mlir {
namespace heir {
namespace lwe {

namespace {
template <typename OpTy>
struct LweCiphertextPlaintextOpPlaintextOperandImpl
    : public ::mlir::heir::PlaintextOperandInterface::ExternalModel<
          LweCiphertextPlaintextOpPlaintextOperandImpl<OpTy>, OpTy> {
  SmallVector<unsigned> maybePlaintextOperands(Operation* op) const {
    SmallVector<unsigned> indices;
    for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<LWEPlaintextType>(getElementTypeOrSelf(operand.getType()))) {
        indices.push_back(i);
      }
    }
    return indices;
  }
};
}  // namespace

void LWEDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/LWE/IR/LWEAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/LWE/IR/LWETypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/LWE/IR/LWEOps.cpp.inc"
      >();

  RMulPlainOp::attachInterface<
      LweCiphertextPlaintextOpPlaintextOperandImpl<RMulPlainOp>>(*getContext());
  RAddPlainOp::attachInterface<
      LweCiphertextPlaintextOpPlaintextOperandImpl<RAddPlainOp>>(*getContext());
  RSubPlainOp::attachInterface<
      LweCiphertextPlaintextOpPlaintextOperandImpl<RSubPlainOp>>(*getContext());
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
