
#include <ostream>

#include "llvm/include/llvm/Support/raw_os_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {
namespace heir {

// A struct to hold a perfect loop nest, including a list of induction
// variables and corresponding upper and lower bounds.
struct LoopNest {
  unsigned numInductionVars;

  // Lower and upper bounds for each induction variable.
  std::vector<int64_t> lowerBounds;
  std::vector<int64_t> upperBounds;

  // Constraints for the body of the loop nest
  std::vector<AffineExpr> constraints;
  // Whether each constraint above is an equality of inequality; true means
  // equality.
  std::vector<bool> eq;

  bool operator==(const LoopNest& other) const {
    return numInductionVars == other.numInductionVars &&
           lowerBounds == other.lowerBounds &&
           upperBounds == other.upperBounds && constraints == other.constraints;
  }
};

inline std::ostream& operator<<(std::ostream& os, const LoopNest& x) {
  os << "LoopNest(numIterationVars=" << x.numInductionVars << ", lowerBounds=[";
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
  for (auto condition : x.constraints) {
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
//
// This is intended to support the case where the IntegerRelation defines a
// single polyhedron representing a ciphertext layout, and the code generated
// for the packing can be expressed as a single perfect loop nest with a single
// assignment operator in the innermost loop body.
FailureOr<LoopNest> generateLoopNest(const presburger::IntegerRelation& rel,
                                     MLIRContext* context);

}  // namespace heir
}  // namespace mlir
