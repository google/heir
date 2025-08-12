#include "lib/Utils/Layout/Parser.h"

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

namespace mlir {
namespace heir {

using presburger::IntegerRelation;
using presburger::VarKind;

presburger::IntegerRelation relationFromString(StringRef integerSetStr,
                                               int numDomainVars,
                                               MLIRContext* context) {
  IntegerRelation relation = affine::FlatAffineValueConstraints(
      parseIntegerSet(integerSetStr, context));
  relation.convertVarKind(VarKind::SetDim, 0, numDomainVars, VarKind::Domain);
  return relation;
}

}  // namespace heir
}  // namespace mlir
