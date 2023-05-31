#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineValueMap.h" // from @llvm-project

namespace mlir {

namespace heir {

// getFlattenedAccessIndex gets the flattened access index for MemRef access
// given the MemRef type's shape. Returns a std::nullopt if the indices are not
// constants (e.g. derived from inputs).
std::optional<uint64_t> getFlattenedAccessIndex(affine::MemRefAccess access,
                                                mlir::Type memRefType) {
  affine::AffineValueMap thisMap;
  access.getAccessMap(&thisMap);
  std::vector<uint64_t> accessIndices;
  for (auto i = 0; i < access.getRank(); ++i) {
    // The access indices of the global memref *must* be constant,
    // meaning that they cannot be a variable access (for example, a
    // loop index) or symbolic, for example, an input symbol.
    if (thisMap.getResult(i).getKind() != AffineExprKind::Constant) {
      return std::nullopt;
    }
    accessIndices.push_back(
        (thisMap.getResult(i).dyn_cast<mlir::AffineConstantExpr>()).getValue());
  }

  return mlir::ElementsAttr::getFlattenedIndex(
      memRefType, llvm::ArrayRef<uint64_t>(accessIndices));
}

}  // namespace heir

}  // namespace mlir
