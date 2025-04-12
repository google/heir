#include "lib/Dialect/ModArith/IR/ModArithTypes.h"

#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace mod_arith {

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
