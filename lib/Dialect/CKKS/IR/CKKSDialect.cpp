#include "lib/Dialect/CKKS/IR/CKKSDialect.h"

// IWYU pragma: begin_keep
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
// IWYU pragma: end_keep

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"

// Generated definitions
#include "lib/Dialect/CKKS/IR/CKKSDialect.cpp.inc"
#include "mlir/include/mlir/IR/Operation.h"      // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/CKKS/IR/CKKSAttributes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/CKKS/IR/CKKSOps.cpp.inc"

namespace mlir {
namespace heir {
namespace ckks {

namespace {
template <typename OpTy>
struct CkksCiphertextPlaintextOpPlaintextOperandImpl
    : public ::mlir::heir::PlaintextOperandInterface::ExternalModel<
          CkksCiphertextPlaintextOpPlaintextOperandImpl<OpTy>, OpTy> {
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
// CKKS dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void CKKSDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/CKKS/IR/CKKSAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/CKKS/IR/CKKSOps.cpp.inc"
      >();

  AddPlainOp::attachInterface<
      CkksCiphertextPlaintextOpPlaintextOperandImpl<AddPlainOp>>(*getContext());
  SubPlainOp::attachInterface<
      CkksCiphertextPlaintextOpPlaintextOperandImpl<SubPlainOp>>(*getContext());
  MulPlainOp::attachInterface<
      CkksCiphertextPlaintextOpPlaintextOperandImpl<MulPlainOp>>(*getContext());
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
