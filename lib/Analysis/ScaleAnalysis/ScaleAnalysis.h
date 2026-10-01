#ifndef LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_
#define LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_

#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Parameters/CKKS/Params.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

template <typename ScaleModelT>
class ScaleState {
 public:
  ScaleState() : scale(std::nullopt) {}
  explicit ScaleState(int64_t scale) : scale(scale) {}

  int64_t getScale() const {
    assert(isInitialized());
    return scale.value();
  }

  bool operator==(const ScaleState& rhs) const { return scale == rhs.scale; }

  bool isInitialized() const { return scale.has_value(); }

  static ScaleState join(const ScaleState& lhs, const ScaleState& rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    return ScaleState(
        ScaleModelT::evalAddScale({lhs.getScale(), rhs.getScale()}));
  }

  static ScaleState meet(const ScaleState& lhs, const ScaleState& rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;

    return ScaleState(
        ScaleModelT::evalAddScale({lhs.getScale(), rhs.getScale()}));
  }

  void print(llvm::raw_ostream& os) const {
    if (isInitialized()) {
      os << "ScaleState(" << scale.value() << ")";
    } else {
      os << "ScaleState(uninitialized)";
    }
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const ScaleState<ScaleModelT>& state) {
    state.print(os);
    return os;
  }

 private:
  // This may not represent 2 ** 80 scale for CKKS.
  // Currently we use logScale for CKKS.
  std::optional<int64_t> scale;
};

template <typename ScaleModelT>
class ScaleLattice : public dataflow::Lattice<ScaleState<ScaleModelT>> {
 public:
  using dataflow::Lattice<ScaleState<ScaleModelT>>::Lattice;
};

struct BGVScaleModel {
  using SchemeParam = bgv::SchemeParam;
  using LocalParam = bgv::LocalParam;

  static int64_t evalAddScale(ArrayRef<int64_t> scales);
  static int64_t evalMulScale(const LocalParam& param, int64_t lhs,
                              int64_t rhs);
  static int64_t evalMulScaleBackward(const LocalParam& param, int64_t result,
                                      int64_t lhs);
  static int64_t evalModReduceScale(const LocalParam& inputParam,
                                    int64_t scale);
  static int64_t evalModReduceScaleBackward(const LocalParam& inputParam,
                                            int64_t resultScale);
  static std::optional<int64_t> getDefaultScale(const SchemeParam& param);
  static int64_t evalMulTargetScale(const LocalParam& param);
};

struct CKKSScaleModel {
  using SchemeParam = ckks::SchemeParam;
  using LocalParam = ckks::LocalParam;

  static int64_t evalAddScale(ArrayRef<int64_t> scales);
  static int64_t evalMulScale(const LocalParam& param, int64_t lhs,
                              int64_t rhs);

  static int64_t evalModReduceScale(const LocalParam& inputParam,
                                    int64_t scale);

  static std::optional<int64_t> getDefaultScale(const SchemeParam& param);
  static int64_t evalMulTargetScale(const LocalParam& param);
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
    : public dataflow::SparseForwardDataFlowAnalysis<ScaleLattice<ScaleModelT>>,
      public SecretnessAnalysisDependent<ScaleAnalysis<ScaleModelT>> {
 public:
  using dataflow::SparseForwardDataFlowAnalysis<
      ScaleLattice<ScaleModelT>>::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<ScaleAnalysis<ScaleModelT>>;

  using SchemeParamType = typename ScaleModelT::SchemeParam;
  using LocalParamType = typename ScaleModelT::LocalParam;

  ScaleAnalysis(DataFlowSolver& solver, const SchemeParamType& schemeParam,
                int64_t inputScale, bool assumeTargetScaleForMul = false)
      : dataflow::SparseForwardDataFlowAnalysis<ScaleLattice<ScaleModelT>>(
            solver),
        schemeParam(schemeParam),
        inputScale(inputScale),
        assumeTargetScaleForMul(assumeTargetScaleForMul) {}

  void setToEntryState(ScaleLattice<ScaleModelT>* lattice) override {
    if (isa<secret::SecretType>(lattice->getAnchor().getType())) {
      this->propagateIfChanged(
          lattice, lattice->join(ScaleState<ScaleModelT>(inputScale)));
      return;
    }
    this->propagateIfChanged(lattice, lattice->join(ScaleState<ScaleModelT>()));
  }

  LogicalResult visitOperation(
      Operation* op, ArrayRef<const ScaleLattice<ScaleModelT>*> operands,
      ArrayRef<ScaleLattice<ScaleModelT>*> results) override;

  void visitExternalCall(
      CallOpInterface call,
      ArrayRef<const ScaleLattice<ScaleModelT>*> argumentLattices,
      ArrayRef<ScaleLattice<ScaleModelT>*> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState* state, ChangeResult changed) {
    this->propagateIfChanged(state, changed);
  }

 private:
  const SchemeParamType schemeParam;
  int64_t inputScale;
  bool assumeTargetScaleForMul;
};

template <typename ScaleModelT>
class ScaleAnalysisBackward
    : public dataflow::SparseBackwardDataFlowAnalysis<
          ScaleLattice<ScaleModelT>>,
      public SecretnessAnalysisDependent<ScaleAnalysisBackward<ScaleModelT>> {
 public:
  using dataflow::SparseBackwardDataFlowAnalysis<
      ScaleLattice<ScaleModelT>>::SparseBackwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<ScaleAnalysisBackward<ScaleModelT>>;

  using SchemeParamType = typename ScaleModelT::SchemeParam;
  using LocalParamType = typename ScaleModelT::LocalParam;

  ScaleAnalysisBackward(DataFlowSolver& solver,
                        SymbolTableCollection& symbolTable,
                        const SchemeParamType& schemeParam)
      : dataflow::SparseBackwardDataFlowAnalysis<ScaleLattice<ScaleModelT>>(
            solver, symbolTable),
        schemeParam(schemeParam) {}

  void setToExitState(ScaleLattice<ScaleModelT>* lattice) override {
    Value val = lattice->getAnchor();
    auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(val);
    if (mgmtAttr && mgmtAttr.getScale() != -1) {
      this->propagateIfChanged(
          lattice, lattice->join(ScaleState<ScaleModelT>(mgmtAttr.getScale())));
      return;
    }
    this->propagateIfChanged(lattice, lattice->join(ScaleState<ScaleModelT>()));
  }

  LogicalResult visitOperation(
      Operation* op, ArrayRef<ScaleLattice<ScaleModelT>*> operands,
      ArrayRef<const ScaleLattice<ScaleModelT>*> results) override;

  // dummy impl
  void visitBranchOperand(OpOperand& operand) override {}
  void visitCallOperand(OpOperand& operand) override {}
  void visitNonControlFlowArguments(
      RegionSuccessor& successor, ArrayRef<BlockArgument> arguments) override {}

 private:
  const SchemeParamType schemeParam;
};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

template <typename ScaleModelT>
int64_t getScale(Value value, DataFlowSolver* solver);

constexpr StringRef kArgScaleAttrName = "mgmt.scale";

template <typename ScaleModelT>
void annotateScale(Operation* top, DataFlowSolver* solver);

int64_t getScaleFromMgmtAttr(Value value);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_
