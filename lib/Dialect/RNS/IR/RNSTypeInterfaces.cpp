#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"

#include <cstddef>

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "llvm/include/llvm/ADT/APInt.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {

using mod_arith::ModArithDialect;
using mod_arith::ModArithType;

namespace rns {

struct ModArithRNSBasisTypeInterface
    : public RNSBasisTypeInterface::ExternalModel<ModArithRNSBasisTypeInterface,
                                                  ModArithType> {
  bool isCompatibleWith(Type type, Type otherRnsBasisType) const {
    auto thisType = mlir::dyn_cast<ModArithType>(type);
    if (!thisType) {
      return false;
    }

    auto other = mlir::dyn_cast<ModArithType>(otherRnsBasisType);
    if (!other) {
      return false;
    }

    auto thisStorageType = thisType.getModulus().getType();
    auto otherStorageType = other.getModulus().getType();
    APInt thisModulus = thisType.getModulus().getValue();
    APInt otherModulus = other.getModulus().getValue();

    // require same storage type
    if (thisStorageType != otherStorageType) {
      return false;
    }

    // coprime test
    return llvm::APIntOps::GreatestCommonDivisor(thisModulus, otherModulus) ==
           1;
  }
};

void registerExternalRNSTypeInterfaces(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, ModArithDialect* dialect) {
    ModArithType::attachInterface<ModArithRNSBasisTypeInterface>(*ctx);
  });
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
