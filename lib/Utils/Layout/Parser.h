// Helper functions to parse an integer relation from a string

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {
namespace heir {

presburger::IntegerRelation relationFromString(llvm::StringRef integerSetStr,
                                               int numDomainVars,
                                               MLIRContext *context);

}  // namespace heir
}  // namespace mlir
