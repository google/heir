#include "lib/Dialect/RNS/IR/RNSTypes.h"

#include <cstddef>

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
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
using polynomial::PolynomialType;

namespace rns {

LogicalResult RNSType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<mlir::Type> basisTypes) {
  bool compatible = true;

  auto getInterface = [&](Type type) -> FailureOr<RNSBasisTypeInterface> {
    auto res = mlir::dyn_cast<RNSBasisTypeInterface>(type);
    if (!res) {
      return emitError() << type << " does not have RNSBasisTypeInterface";
    }
    return res;
  };

  size_t numTypes = basisTypes.size();
  for (auto i = 0; i != numTypes; ++i) {
    for (auto j = i + 1; j != numTypes; ++j) {
      auto resI = getInterface(basisTypes[i]);
      if (failed(resI)) {
        return resI;
      }
      auto resJ = getInterface(basisTypes[j]);
      if (failed(resJ)) {
        return resJ;
      }
      compatible &= (*resI).isCompatibleWith(*resJ);
    }
  }

  if (!compatible) {
    return emitError() << "RNS type has incompatible basis types";
  }
  return success();
}

struct PolynomialRNSBasisTypeInterface
    : public RNSBasisTypeInterface::ExternalModel<
          PolynomialRNSBasisTypeInterface, PolynomialType> {
  bool isCompatibleWith(Type type, Type otherRnsBasisType) const {
    auto thisType = mlir::dyn_cast<PolynomialType>(type);
    if (!thisType) {
      return false;
    }

    auto other = mlir::dyn_cast<PolynomialType>(otherRnsBasisType);
    if (!other) {
      return false;
    }

    // The coefficient moduli may be different, but the polynomial moduli must
    // agree. This is the typical RNS situation where the point is to avoid
    // using big-integer coefficient moduli by converting them to a smaller set
    // of prime moduli.
    return thisType.getRing().getPolynomialModulus() ==
           other.getRing().getPolynomialModulus();
  }
};

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

void registerExternalRNSTypeInterfaces(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx,
          ::mlir::heir::polynomial::PolynomialDialect *dialect) {
        PolynomialType::attachInterface<PolynomialRNSBasisTypeInterface>(*ctx);
      });
  registry.addExtension(+[](MLIRContext *ctx, ModArithDialect *dialect) {
    ModArithType::attachInterface<ModArithRNSBasisTypeInterface>(*ctx);
  });
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
