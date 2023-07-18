#ifndef HEIR_INCLUDE_DIALECT_POLY_IR_POLYNOMIALDETAIL_H_
#define HEIR_INCLUDE_DIALECT_POLY_IR_POLYNOMIALDETAIL_H_

#include "include/Dialect/Poly/IR/Polynomial.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/Support/TrailingObjects.h"  // from @llvm-project
#include "mlir/include/mlir/Support/StorageUniquer.h"   // from @llvm-project
#include "mlir/include/mlir/Support/TypeID.h"           // from @llvm-project

namespace mlir {
namespace heir {
namespace poly {
namespace detail {

// A Polynomial is stored as an ordered list of monomial terms, each of which
// is a tuple of coefficient and exponent.
struct PolynomialStorage final
    : public StorageUniquer::BaseStorage,
      public llvm::TrailingObjects<PolynomialStorage, Monomial> {
  /// The hash key used for uniquing.
  using KeyTy = std::tuple<unsigned, ArrayRef<Monomial>>;

  unsigned numTerms;

  MLIRContext *context;

  /// The monomial terms for this polynomial.
  ArrayRef<Monomial> terms() const {
    return {getTrailingObjects<Monomial>(), numTerms};
  }

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == numTerms && std::get<1>(key) == terms();
  }

  // Constructs a PolynomialStorage from a key. The context must be set by the
  // caller.
  static PolynomialStorage *construct(
      StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto terms = std::get<1>(key);
    auto byteSize = PolynomialStorage::totalSizeToAlloc<Monomial>(terms.size());
    auto *rawMem = allocator.allocate(byteSize, alignof(PolynomialStorage));
    auto *res = new (rawMem) PolynomialStorage();
    res->numTerms = std::get<0>(key);
    std::uninitialized_copy(terms.begin(), terms.end(),
                            res->getTrailingObjects<Monomial>());
    return res;
  }
};

}  // namespace detail
}  // namespace poly
}  // namespace heir
}  // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::heir::poly::detail::PolynomialStorage)

#endif  // HEIR_INCLUDE_DIALECT_POLY_IR_POLYNOMIALDETAIL_H_
