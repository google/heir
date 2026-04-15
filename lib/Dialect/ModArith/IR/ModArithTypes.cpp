#include "lib/Dialect/ModArith/IR/ModArithTypes.h"

#include "llvm/include/llvm/ADT/APInt.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace mod_arith {

#include "lib/Dialect/ModArith/IR/ModArithTypeInterfaces.cpp.inc"

Type ModArithType::getLoweringType() const { return getModulus().getType(); }

Type ModArithType::getResidueType(unsigned index) const {
  if (index != 0) {
    return Type();
  }
  return *this;
}

bool ModArithType::isCompatibleWith(Type otherRnsBasisType) const {
  auto other = mlir::dyn_cast<ModArithType>(otherRnsBasisType);
  if (!other) {
    return false;
  }

  auto thisStorageType = getModulus().getType();
  auto otherStorageType = other.getModulus().getType();
  APInt thisModulus = getModulus().getValue();
  APInt otherModulus = other.getModulus().getValue();

  if (thisStorageType != otherStorageType) {
    return false;
  }

  return llvm::APIntOps::GreatestCommonDivisor(thisModulus, otherModulus) == 1;
}

LogicalResult ModArithType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::IntegerAttr modulus) {
  APInt value = modulus.getValue();
  unsigned bitWidth = value.getBitWidth();
  unsigned modWidth = value.getActiveBits();
  if (modWidth > bitWidth - 1)
    return emitError()
           << "underlying type's bitwidth must be 1 bit larger than "
           << "the modulus bitwidth, but got " << bitWidth
           << " while modulus requires width " << modWidth << ".";
  return success();
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
