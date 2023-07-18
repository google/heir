#include "include/Dialect/Poly/IR/Polynomial.h"

#include "include/Dialect/Poly/IR/PolynomialDetail.h"
#include "llvm/include/llvm/ADT/APInt.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallString.h"      // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"       // from @llvm-project

namespace mlir {
namespace heir {
namespace poly {

MLIRContext *Polynomial::getContext() const { return terms->context; }

ArrayRef<Monomial> Polynomial::getTerms() const { return terms->terms(); }

Polynomial Polynomial::fromMonomials(ArrayRef<Monomial> monomials,
                                     MLIRContext *context) {
  auto assignCtx = [context](detail::PolynomialStorage *storage) {
    storage->context = context;
  };

  // A polynomial's terms are canonically stored in order of increasing degree.
  llvm::OwningArrayRef<Monomial> monomials_copy =
      llvm::OwningArrayRef<Monomial>(monomials);
  std::sort(monomials_copy.begin(), monomials_copy.end());

  StorageUniquer &uniquer = context->getAttributeUniquer();
  return Polynomial(uniquer.get<detail::PolynomialStorage>(
      assignCtx, monomials.size(), monomials_copy));
}

Polynomial Polynomial::fromCoefficients(ArrayRef<int64_t> coeffs,
                                        MLIRContext *context) {
  std::vector<Monomial> monomials;
  for (size_t i = 0; i < coeffs.size(); i++) {
    monomials.push_back(Monomial(coeffs[i], i));
  }
  return Polynomial::fromMonomials(std::move(monomials), context);
}

void Polynomial::print(raw_ostream &os) const {
  bool first = true;
  for (auto term : terms->terms()) {
    if (first) {
      first = false;
    } else {
      os << " + ";
    }
    std::string coeff_to_print;
    if (term.coefficient == 1 && term.exponent.uge(1)) {
      coeff_to_print = "";
    } else {
      llvm::SmallString<512> coeff_string;
      term.coefficient.toStringSigned(coeff_string);
      coeff_to_print = coeff_string.str();
    }

    if (term.exponent == 0) {
      os << coeff_to_print;
    } else if (term.exponent == 1) {
      os << coeff_to_print << "x";
    } else {
      os << coeff_to_print << "x**" << term.exponent;
    }
  }
}

}  // end namespace poly
}  // end namespace heir
}  // end namespace mlir

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::heir::poly::detail::PolynomialStorage);
