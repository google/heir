
#include <ostream>

#include "llvm/include/llvm/Support/raw_os_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {
namespace heir {

// A struct for a single loop
//
// for (inductionVar = lowerBound; inductionVar < upperBound; ++inductionVar) {
//   insert statements for eliminated variables;
//   check constraints
//   inner loop body
// }
struct Loop {
  unsigned inductionVar;  // Index of the induction variable in the relation
  int64_t lowerBound;     // Lower bound for the induction variable
  int64_t upperBound;     // Upper bound for the induction variable

  // Variables in scope of this loop, including the new induction variable.
  std::vector<unsigned> scope;

  // Expressions to write variables in terms of the current scope.
  std::unordered_map<unsigned, AffineExpr> eliminatedVariables;

  // Constraints for the body of this loop.
  std::vector<AffineExpr> constraints;

  // Whether each constraint above is an equality of inequality; true means
  // equality. The other side of the equality or inequality is always zero.
  std::vector<bool> eq;

  bool operator==(const Loop& other) const {
    return inductionVar == other.inductionVar &&
           lowerBound == other.lowerBound && upperBound == other.upperBound &&
           scope == other.scope &&
           eliminatedVariables == other.eliminatedVariables && eq == other.eq;

    // For some reason the AffineExpr equality operator doesn't work
    // because the first half-byte is off. Sometimes applying simplifyAffineExpr
    // fixes it, but not always. So constraints are not compared here.
    //
    // && constraints == other.constraints;
  }
};

// A struct to hold a perfect loop nest.
struct LoopNest {
  SmallVector<Loop, 4> loops;

  Loop* addLoop(unsigned varIndex, int64_t lowerBound, int64_t upperBound) {
    Loop loop;
    loop.inductionVar = varIndex;
    loop.lowerBound = lowerBound;
    loop.upperBound = upperBound;

    if (!loops.empty()) {
      // copy the scope and eliminatedVariables from the innermost loop to the
      // new loop
      loop.scope = loops.back().scope;
    }

    loop.scope.push_back(varIndex);
    loops.push_back(std::move(loop));
    return &loops.back();
  }

  bool operator==(const LoopNest& other) const { return loops == other.loops; }
};

inline std::ostream& operator<<(std::ostream& stdOs, const Loop& x) {
  llvm::raw_os_ostream os(stdOs);
  os << "Loop(varIndex=" << x.inductionVar << ", lowerBound=" << x.lowerBound
     << ", upperBound=" << x.upperBound << ", scope=[";
  for (size_t i = 0; i < x.scope.size(); ++i) {
    os << x.scope[i];
    if (i < x.scope.size() - 1) {
      os << ", ";
    }
  }
  os << "], eliminatedVariables={";
  for (const auto& [var, expr] : x.eliminatedVariables) {
    os << var << ": " << expr << ", ";
  }
  os << "}, constraints=[";
  for (size_t i = 0; i < x.constraints.size(); ++i) {
    x.constraints[i].print(os);
    if (i < x.constraints.size() - 1) {
      os << ", ";
    }
  }
  os << "], eq=[";
  for (size_t i = 0; i < x.eq.size(); ++i) {
    os << (x.eq[i] ? "true" : "false");
    if (i < x.eq.size() - 1) {
      os << ", ";
    }
  }
  os << "])";
  os.flush();
  return stdOs;
}

inline std::ostream& operator<<(std::ostream& os, const LoopNest& x) {
  os << "LoopNest(loops=[";
  for (size_t i = 0; i < x.loops.size(); ++i) {
    os << x.loops[i];
    if (i < x.loops.size() - 1) {
      os << ", ";
    }
  }
  os << "])";
  return os;
}

// For googletest integration
void PrintTo(const Loop& obj, std::ostream* os) { *os << obj; }

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
