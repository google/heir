#ifndef LIB_ANALYSIS_SECRETNESSANALYSIS_SECRETNESSANALYSIS_H_
#define LIB_ANALYSIS_SECRETNESSANALYSIS_SECRETNESSANALYSIS_H_

#include <cassert>
#include <optional>

#include "lib/Dialect/HEIRInterfaces.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"               // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

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

  const bool& getSecretness() const {
    assert(isInitialized());
    return *secretness;
  }
  const bool& get() const { return getSecretness(); }
  void setSecretness(bool value) { secretness = value; }

  // Check if the Secretness state is initialized. It can be uninitialized if
  // the state hasn't yet been set during the analysis.
  bool isInitialized() const { return secretness.has_value(); }

  bool operator==(const Secretness& rhs) const {
    return secretness == rhs.secretness;
  }

  // Join two Secretness states
  static Secretness join(const Secretness& lhs, const Secretness& rhs) {
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

    for (const auto& secretness : secretnesses) {
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

  void print(raw_ostream& os) const { os << "Secretness: " << secretness; }

 private:
  // Stores the Secretness state if known
  std::optional<bool> secretness;
};

inline raw_ostream& operator<<(raw_ostream& os, const Secretness& value) {
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
  explicit SecretnessAnalysis(DataFlowSolver& solver)
      : SparseForwardDataFlowAnalysis(solver) {}
  ~SecretnessAnalysis() override = default;
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  // Set Secretness state for an initialized SSA value
  void setToEntryState(SecretnessLattice* lattice) override;
  // Set Secretness state for SSA value(s) produced by an operation
  LogicalResult visitOperation(Operation* operation,
                               ArrayRef<const SecretnessLattice*> operands,
                               ArrayRef<SecretnessLattice*> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const SecretnessLattice*> argumentLattices,
                         ArrayRef<SecretnessLattice*> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState* state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }
};

/**
 * @class SecretnessAnalysisDependent
 * @brief A class that provides methods to analyze and ensure the secretness of
 * operations and their operands/results.
 *
 * This class is designed to be used within an analysis framework to determine
 * the secretness of values and create dependencies on SecretnessAnalysis.
 */
template <typename ChildAnalysis>
class SecretnessAnalysisDependent {
 private:
  ChildAnalysis* getChildAnalysis() {
    return static_cast<ChildAnalysis*>(this);
  }

 protected:
  /**
   * @brief Ensures the secretness of a given value within an operation.
   *
   * This method creates a SecretnessLattice for the given value and establishes
   * a dependency on SecretnessAnalysis.
   *
   * @param op The operation containing the value (either Operand or Result).
   * @param value The value to check for secretness.
   * @return true if the value is secret, false if the secretness of the value
   * is unknown or false.
   */
  bool isSecretInternal(Operation* op, Value value) {
    // create dependency on SecretnessAnalysis
    auto* lattice =
        getChildAnalysis()->template getOrCreateFor<SecretnessLattice>(
            getChildAnalysis()->getProgramPointAfter(op), value);
    if (!lattice->getValue().isInitialized()) {
      return false;
    }
    return lattice->getValue().getSecretness();
  };

  /**
   * @brief Selects the OpResults of an operation that are secret (secretness =
   * true).
   *
   * This method iterates through the results of the given operation and adds
   * those that are secret to the provided vector.
   *
   * @param op The operation to analyze.
   * @param secretResults A vector to store the secret results.
   */
  void getSecretResults(Operation* op,
                        SmallVectorImpl<OpResult>& secretResults) {
    for (const auto& result : op->getOpResults()) {
      if (isSecretInternal(op, result)) {
        secretResults.push_back(result);
      }
    }
  }

  /**
   * @brief Selects the indices of OpResults of an operation that are secret
   * (secretness = true).
   */
  void getSecretResultIndices(Operation* op,
                              SmallVectorImpl<unsigned>& secretResults) {
    for (const auto& [i, result] : llvm::enumerate(op->getOpResults())) {
      if (isSecretInternal(op, result)) {
        secretResults.push_back(i);
      }
    }
  }

  /**
   * @brief Selects the OpOperands of an operation that are secret (secretness =
   * true).
   *
   * This method iterates through the operands of the given operation and adds
   * those that are secret to the provided vector.
   *
   * @param op The operation to analyze.
   * @param secretOperands A vector to store the secret operands.
   */
  void getSecretOperands(Operation* op,
                         SmallVectorImpl<OpOperand*>& secretOperands) {
    for (auto& operand : op->getOpOperands()) {
      if (isSecretInternal(op, operand.get())) {
        secretOperands.push_back(&operand);
      }
    }
  }

  /**
   * @brief Selects the OpOperands of an operation that are not secret
   * (secretness = false or unknown), but may be plaintexts.
   *
   * The input operation must either have no legal plaintext operands, or else
   * implement PlaintextOperandInterface to identify which operands may be
   * plaintext. Those are then filtered by secretness to populate the outparam.
   *
   * @param op The operation to analyze.
   * @param plaintextOperands A vector to store the non-secret operands.
   */
  void getPlaintextOperands(Operation* op,
                            SmallVectorImpl<OpOperand*>& plaintextOperands) {
    auto opInterface = dyn_cast<PlaintextOperandInterface>(op);
    if (opInterface) {
      SmallVector<unsigned> maybePlaintextOperands =
          opInterface.maybePlaintextOperands();
      for (unsigned operandIndex : maybePlaintextOperands) {
        if (!isSecretInternal(op, op->getOperand(operandIndex))) {
          plaintextOperands.push_back(&op->getOpOperand(operandIndex));
        }
      }
    }
  }
};

// Annotate the secretness of operation based on the secretness of its results
// If verbose = true, annotates the secretness of *all* values,
// including ones with public secretness , missing, or inconclusive analysis.
void annotateSecretness(Operation* top, DataFlowSolver* solver, bool verbose);

// this method is used when DataFlowSolver has finished running the secretness
// analysis
bool isSecret(Value value, DataFlowSolver* solver);

bool isSecret(const SecretnessLattice* lattice);

bool isSecret(ValueRange values, DataFlowSolver* solver);

void getSecretOperands(Operation* op,
                       SmallVectorImpl<OpOperand*>& secretOperands,
                       DataFlowSolver* solver);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SECRETNESSANALYSIS_SECRETNESSANALYSIS_H_
