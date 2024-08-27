#ifndef LIB_ANALYSIS_SECRETNESSANALYSIS_SECRETNESSANALYSIS_H_
#define LIB_ANALYSIS_SECRETNESSANALYSIS_SECRETNESSANALYSIS_H_

#include <optional>

#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project

namespace mlir {
namespace heir {

// Secretness describes if an SSA value is of secret type or not. An SSA value
// is considered to be of secret type if one of the following is true:
//      (1) the value defined is wrapped by the secret.secret type
//      (2) the value is produced by an operation that uses an operand of
//      secret.secret type
class Secretness {
 public:
  Secretness() : secretness(std::nullopt) {}
  explicit Secretness(bool value) : secretness(value) {}
  ~Secretness() = default;

  const bool &getSecretness() const {
    assert(isInitialized());
    return *secretness;
  }
  void setSecretness(bool value) { secretness = value; }

  // Check if the Secretness state is initialized. It can be uninitialized if
  // the state hasn't yet been set during the analysis.
  bool isInitialized() const { return secretness.has_value(); }

  bool operator==(const Secretness &rhs) const {
    return secretness == rhs.secretness;
  }

  // Join two Secretness states
  static Secretness join(const Secretness &lhs, const Secretness &rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    if (lhs.getSecretness() == rhs.getSecretness()) return lhs;

    return Secretness{lhs.getSecretness() || rhs.getSecretness()};
  }

  static Secretness combine(llvm::ArrayRef<Secretness> secretnesses) {
    // initialize secretness to false
    Secretness result = Secretness(false);

    // Assume every secretness state is initialized/known
    bool uninitializedFound = false;

    for (const auto &secretness : secretnesses) {
      // If uninitialized state is found, set to true
      if (!secretness.isInitialized())
        uninitializedFound = true;
      else {
        // If any element in the list is secret, the combination is also
        // considered secret
        if (secretness.getSecretness()) return secretness;
      }
    }
    // If execution reaches here and an uninitialized (unknown) secretness is
    // found, unknown combined with Secretness(false) will be unknown
    if (uninitializedFound) return Secretness();

    return result;
  }

  void print(raw_ostream &os) const { os << "Secretness: " << secretness; }

 private:
  // Stores the Secretness state if known
  std::optional<bool> secretness;
};

inline raw_ostream &operator<<(raw_ostream &os, const Secretness &value) {
  value.print(os);
  return os;
}
class SecretnessLattice : public dataflow::Lattice<Secretness> {
 public:
  using Lattice::Lattice;
};

// An analysis that identifies and assigns the Secretness of an SSA value in a
// program. This is used by other passes to selectively apply transformations on
// operations that evaluate secret types. We use a forward dataflow analysis
// because the Secretness state propagates forward from the input arguments of a
// function down to values produced from operations that use these arguments.

class SecretnessAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<SecretnessLattice> {
 public:
  explicit SecretnessAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis(solver) {}
  ~SecretnessAnalysis() override = default;
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  // Set Secretness state for an initialized SSA value
  void setToEntryState(SecretnessLattice *lattice) override;
  // Set Secretness state for SSA value(s) produced by an operation
  LogicalResult visitOperation(Operation *operation,
                               ArrayRef<const SecretnessLattice *> operands,
                               ArrayRef<SecretnessLattice *> results) override;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SECRETNESSANALYSIS_SECRETNESSANALYSIS_H_
