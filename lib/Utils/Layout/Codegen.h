
#include <ostream>

#include "llvm/include/llvm/Support/raw_os_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {
namespace heir {

// A struct to hold a perfect loop nest, including a list of iteration
// variables and corresponding upper and lower bounds.
//
// The final member is an IntegerSet that corresponds to the condition to check
// on the iteration variables to determine if the loop body should execute.
struct LoopNest {
  unsigned numIterationVars;
  std::vector<int64_t> lowerBounds;
  std::vector<int64_t> upperBounds;
  std::vector<AffineExpr> conditions;

  bool operator==(const LoopNest& other) const {
    return numIterationVars == other.numIterationVars &&
           lowerBounds == other.lowerBounds &&
           upperBounds == other.upperBounds && conditions == other.conditions;
  }
};

inline std::ostream& operator<<(std::ostream& os, const LoopNest& x) {
  os << "LoopNest(numIterationVars=" << x.numIterationVars << ", lowerBounds=[";
  for (size_t i = 0; i < x.lowerBounds.size(); ++i) {
    os << x.lowerBounds[i];
    if (i < x.lowerBounds.size() - 1) {
      os << ", ";
    }
  }
  os << "], upperBounds=[";
  for (size_t i = 0; i < x.upperBounds.size(); ++i) {
    os << x.upperBounds[i];
    if (i < x.upperBounds.size() - 1) {
      os << ", ";
    }
  }

  os << "], conditions=[";
  llvm::raw_os_ostream llvmOs(os);
  for (auto condition : x.conditions) {
    condition.print(llvmOs);
    llvmOs << ", ";
  }
  llvmOs.flush();
  os << ")";
  return os;
}

// For googletest integration
void PrintTo(const LoopNest& obj, std::ostream* os) { *os << obj; }

// Generate a LoopNest from an IntegerRelation.
FailureOr<LoopNest> generateLoopNest(const presburger::IntegerRelation& rel,
                                     MLIRContext* context);

}  // namespace heir
}  // namespace mlir
