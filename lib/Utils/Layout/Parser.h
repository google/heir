// Helper functions to parse an integer relation from a string

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {
namespace heir {

/// Parse an integer relation from a string representation. A thin wrapper
/// around MLIR's upstream parser for IntegerSet, but in IntegerSet there is no
/// distinction between domain and range dimensions. In IntegerSet, all
/// dimensions are "set" dimensions, and a "set" dimension is interpreted as a
/// "range" dimension in IntegerRelation. In IntegerRelation, all domain
/// dimensions precede all range dimensions, so we need to specify how many of
/// the leading dimensions are domain dimensions.
presburger::IntegerRelation relationFromString(llvm::StringRef integerSetStr,
                                               int numDomainVars,
                                               MLIRContext* context);

}  // namespace heir
}  // namespace mlir
