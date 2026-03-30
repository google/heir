#ifndef LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_H_
#define LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_H_

#include <cstdint>

#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
// IWYU pragma: begin_keep
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "mlir/include/mlir/IR/OpImplementation.h"    // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Rotom/IR/RotomAttributes.h.inc"

namespace mlir::heir::rotom {

enum class LayoutPieceKind { Traversal, Replication, Gap };

struct LayoutData {
  int64_t n;
  int64_t ctPrefixLen;
  llvm::SmallVector<DimAttr> originalDims;
  llvm::SmallVector<DimAttr> traversalDims;
  llvm::SmallVector<DimAttr> replicationDims;
  llvm::SmallVector<DimAttr> gapDims;
  llvm::SmallVector<LayoutPieceKind> pieces;
  llvm::SmallVector<int64_t> pieceIndex;
};

/// Preprocess a Rotom layout.
FailureOr<LayoutData> preprocessLayoutAttr(LayoutAttr attr);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_H_
