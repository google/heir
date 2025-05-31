#ifndef LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_
#define LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_

#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Parameters/PlaintextParams.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

class ScaleState {
 public:
  ScaleState() : scale(std::nullopt) {}
  explicit ScaleState(int64_t scale) : scale(scale) {}

  int64_t getScale() const {
    assert(isInitialized());
    return scale.value();
  }

  bool operator==(const ScaleState &rhs) const { return scale == rhs.scale; }

  bool isInitialized() const { return scale.has_value(); }

  static ScaleState join(const ScaleState &lhs, const ScaleState &rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    // if both are initialized, they should be the same
    return lhs;
  }

  void print(llvm::raw_ostream &os) const {
    if (isInitialized()) {
      os << "ScaleState(" << scale.value() << ")";
    } else {
      os << "ScaleState(uninitialized)";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const ScaleState &state) {
    state.print(os);
    return os;
  }

 private:
  // This may not represent 2 ** 80 scale for CKKS.
  // Currently we use logScale for CKKS.
  std::optional<int64_t> scale;
};

class ScaleLattice : public dataflow::Lattice<ScaleState> {
 public:
  using Lattice::Lattice;
};

struct BGVScaleModel {
  using SchemeParam = bgv::SchemeParam;
  using LocalParam = bgv::LocalParam;

  static int64_t evalMulScale(const LocalParam &param, int64_t lhs,
                              int64_t rhs);
  static int64_t evalMulScaleBackward(const LocalParam &param, int64_t result,
                                      int64_t lhs);
  static int64_t evalModReduceScale(const LocalParam &inputParam,
                                    int64_t scale);
  static int64_t evalModReduceScaleBackward(const LocalParam &inputParam,
                                            int64_t resultScale);
};

struct CKKSScaleModel {
  using SchemeParam = ckks::SchemeParam;
  using LocalParam = ckks::LocalParam;

  static int64_t evalMulScale(const LocalParam &param, int64_t lhs,
                              int64_t rhs);
  static int64_t evalMulScaleBackward(const LocalParam &param, int64_t result,
                                      int64_t lhs);
  static int64_t evalModReduceScale(const LocalParam &inputParam,
                                    int64_t scale);
  static int64_t evalModReduceScaleBackward(const LocalParam &inputParam,
                                            int64_t resultScale);
};

struct PlaintextScaleModel {
  using SchemeParam = PlaintextSchemeParam;
  using LocalParam = PlaintextSchemeParam;

  static int64_t evalMulScale(const LocalParam &param, int64_t lhs,
                              int64_t rhs);
  static int64_t evalMulScaleBackward(const LocalParam &param, int64_t result,
                                      int64_t lhs);
  static int64_t evalModReduceScale(const LocalParam &inputParam,
                                    int64_t scale);
  static int64_t evalModReduceScaleBackward(const LocalParam &inputParam,
                                            int64_t resultScale);
};

/// Forward Analyse the scale of each secret Value
///
/// This forward analysis roots from user input as `inputScale`,
/// and after each HE operation, the scale will be updated.
/// For ct-pt or cross-level operation, we will assume the scale of the
/// undetermined hand side to be the same as the determined one.
/// This forms the level-specific scaling factor constraint.
/// See also the "Ciphertext management" section in the document.
///
/// The analysis will stop propagation for AdjustScaleOp, as the scale
/// of it should be determined together by the forward pass (from input
/// to its operand) and the backward pass (from a determined ciphertext to
/// its result).
///
/// This analysis is expected to determine (almost) all the scales of
/// the secret Value, or ciphertext in the program.
/// The level of plaintext Value, or the opaque result of AdjustLevelOp
/// should be determined by the Backward Analysis below.
template <typename ScaleModelT>
class ScaleAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<ScaleLattice>,
      public SecretnessAnalysisDependent<ScaleAnalysis<ScaleModelT>> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<ScaleAnalysis<ScaleModelT>>;

  using SchemeParamType = typename ScaleModelT::SchemeParam;
  using LocalParamType = typename ScaleModelT::LocalParam;

  ScaleAnalysis(DataFlowSolver &solver, const SchemeParamType &schemeParam,
                int64_t inputScale)
      : dataflow::SparseForwardDataFlowAnalysis<ScaleLattice>(solver),
        schemeParam(schemeParam),
        inputScale(inputScale) {}

  void setToEntryState(ScaleLattice *lattice) override {
    if (isa<secret::SecretType>(lattice->getAnchor().getType())) {
      propagateIfChanged(lattice, lattice->join(ScaleState(inputScale)));
      return;
    }
    propagateIfChanged(lattice, lattice->join(ScaleState()));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const ScaleLattice *> operands,
                               ArrayRef<ScaleLattice *> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const ScaleLattice *> argumentLattices,
                         ArrayRef<ScaleLattice *> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState *state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }

 private:
  const SchemeParamType schemeParam;
  int64_t inputScale;
};

/// Backward Analyse the scale of plaintext Value / opaque result of
/// AdjustLevelOp
///
/// This analysis should be run after the (forward) ScaleAnalysis
/// where the scale of (almost) all the secret Value is determined.
///
/// A special example is ct2 = mul(ct0, rs(adjust_scale(ct1))), where the scale
/// of ct0, ct1, ct2 is determined by the forward pass, rs is rescaling. Then
/// the scale of adjust_scale(ct1) should be determined by the backward pass
/// via backpropagation from ct2 to rs then to adjust_scale.
template <typename ScaleModelT>
class ScaleAnalysisBackward
    : public dataflow::SparseBackwardDataFlowAnalysis<ScaleLattice>,
      public SecretnessAnalysisDependent<ScaleAnalysis<ScaleModelT>> {
 public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<ScaleAnalysis<ScaleModelT>>;

  using SchemeParamType = typename ScaleModelT::SchemeParam;
  using LocalParamType = typename ScaleModelT::LocalParam;

  ScaleAnalysisBackward(DataFlowSolver &solver,
                        SymbolTableCollection &symbolTable,
                        const SchemeParamType &schemeParam)
      : dataflow::SparseBackwardDataFlowAnalysis<ScaleLattice>(solver,
                                                               symbolTable),
        schemeParam(schemeParam) {}

  void setToExitState(ScaleLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(ScaleState()));
  }

  LogicalResult visitOperation(Operation *op, ArrayRef<ScaleLattice *> operands,
                               ArrayRef<const ScaleLattice *> results) override;

  // dummy impl
  void visitBranchOperand(OpOperand &operand) override {}
  void visitCallOperand(OpOperand &operand) override {}

 private:
  const SchemeParamType schemeParam;
};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

int64_t getScale(Value value, DataFlowSolver *solver);

constexpr StringRef kArgScaleAttrName = "mgmt.scale";

void annotateScale(Operation *top, DataFlowSolver *solver);

int64_t getScaleFromMgmtAttr(Value value);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_
