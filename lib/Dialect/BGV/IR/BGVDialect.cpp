#include "lib/Dialect/BGV/IR/BGVDialect.h"

// IWYU pragma: begin_keep
#include <optional>

#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVEnums.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
// IWYU pragma: end_keep

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"

// Generated definitions
#include "lib/Dialect/BGV/IR/BGVDialect.cpp.inc"
#include "lib/Dialect/BGV/IR/BGVEnums.cpp.inc"
#include "mlir/include/mlir/IR/Operation.h"      // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/BGV/IR/BGVAttributes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/BGV/IR/BGVOps.cpp.inc"

namespace mlir {
namespace heir {
namespace bgv {

namespace {
template <typename OpTy>
struct BgvCiphertextPlaintextOpPlaintextOperandImpl
    : public ::mlir::heir::PlaintextOperandInterface::ExternalModel<
          BgvCiphertextPlaintextOpPlaintextOperandImpl<OpTy>, OpTy> {
  SmallVector<unsigned> maybePlaintextOperands(Operation* op) const {
    SmallVector<unsigned> indices;
    for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<lwe::LWEPlaintextType>(getElementTypeOrSelf(operand.getType()))) {
        indices.push_back(i);
      }
    }
    return indices;
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// BGV dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void BGVDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/BGV/IR/BGVAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/BGV/IR/BGVOps.cpp.inc"
      >();

  AddPlainOp::attachInterface<
      BgvCiphertextPlaintextOpPlaintextOperandImpl<AddPlainOp>>(*getContext());
  SubPlainOp::attachInterface<
      BgvCiphertextPlaintextOpPlaintextOperandImpl<SubPlainOp>>(*getContext());
  MulPlainOp::attachInterface<
      BgvCiphertextPlaintextOpPlaintextOperandImpl<MulPlainOp>>(*getContext());
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
