#include "lib/Dialect/RNS/IR/RNSTypes.h"

#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {

using polynomial::PolynomialDialect;
using polynomial::PolynomialType;

namespace heir {
namespace rns {

LogicalResult RNSType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<mlir::Type> basisTypes) {
  bool compatible = true;
  RNSBasisTypeInterface first =
      llvm::dyn_cast<RNSBasisTypeInterface>(basisTypes[0]);
  if (!first) return failure();

  for (auto other : basisTypes) {
    compatible &= first.isCompatibleWith(other);
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

void registerExternalRNSTypeInterfaces(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, ::mlir::polynomial::PolynomialDialect *dialect) {
        PolynomialType::attachInterface<PolynomialRNSBasisTypeInterface>(*ctx);
      });
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
