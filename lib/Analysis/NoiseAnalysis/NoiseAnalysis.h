#ifndef LIB_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_
#define LIB_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_

#include <type_traits>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#ifndef DEBUG_TYPE
#define DEBUG_TYPE "NoiseAnalysis"
#define HEIR_NOISE_ANALYSIS_UNDEF_DEBUG_TYPE
#endif

namespace mlir {
namespace heir {

namespace bfv {
class NoiseByBoundCoeffModel;
class NoiseByVarianceCoeffModel;
class NoiseCanEmbModel;
}  // namespace bfv

template <typename NoiseModel>
struct is_bfv_noise_model : std::false_type {};

template <>
struct is_bfv_noise_model<bfv::NoiseByBoundCoeffModel> : std::true_type {};

template <>
struct is_bfv_noise_model<bfv::NoiseByVarianceCoeffModel> : std::true_type {};

template <>
struct is_bfv_noise_model<bfv::NoiseCanEmbModel> : std::true_type {};

/// This lattice element represents the noise data of an SSA value.
template <typename NoiseState>
class NoiseLattice : public dataflow::Lattice<NoiseState> {
 public:
  using dataflow::Lattice<NoiseState>::Lattice;
};

/// This analysis template takes a noise model as argument and computes the
/// noise data for each SSA value in the program.
template <typename NoiseModelT>
class NoiseAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          NoiseLattice<typename NoiseModelT::StateType>>,
      public SecretnessAnalysisDependent<NoiseAnalysis<NoiseModelT>> {
 public:
  friend class SecretnessAnalysisDependent<NoiseAnalysis<NoiseModelT>>;

  using NoiseModel = NoiseModelT;
  using NoiseState = typename NoiseModelT::StateType;
  using LatticeType = NoiseLattice<NoiseState>;
  using SchemeParamType = typename NoiseModelT::SchemeParamType;
  using LocalParamType = typename NoiseModelT::LocalParamType;

  using dataflow::SparseForwardDataFlowAnalysis<
      LatticeType>::SparseForwardDataFlowAnalysis;

  NoiseAnalysis(DataFlowSolver& solver, const SchemeParamType& schemeParam,
                const NoiseModelT& noiseModel)
      : dataflow::SparseForwardDataFlowAnalysis<LatticeType>(solver),
        schemeParam(schemeParam),
        noiseModel(noiseModel) {}

  void setToEntryState(LatticeType* lattice) override;

  LogicalResult visitOperation(Operation* op,
                               ArrayRef<const LatticeType*> operands,
                               ArrayRef<LatticeType*> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const LatticeType*> argumentLattices,
                         ArrayRef<LatticeType*> resultLattices) override;

 private:
  const SchemeParamType schemeParam;
  const NoiseModelT& noiseModel;
};

template <typename NoiseModelT>
void NoiseAnalysis<NoiseModelT>::setToEntryState(LatticeType* lattice) {
  if (isa<secret::SecretType>(lattice->getAnchor().getType())) {
    Value value = lattice->getAnchor();
    auto localParam =
        LocalParamType(&schemeParam, getLevelFromMgmtAttr(value).getInt(),
                       getDimensionFromMgmtAttr(value));
    NoiseState encrypted = noiseModel.evalEncrypt(localParam);
    this->propagateIfChanged(lattice, lattice->join(encrypted));
    LLVM_DEBUG(llvm::dbgs() << "Initializing "
                            << doubleToString2Prec(
                                   noiseModel.toLogBound(localParam, encrypted))
                            << " to " << value << "\n");
    return;
  }

  this->propagateIfChanged(lattice, lattice->join(NoiseState::uninitialized()));
}

template <typename NoiseModelT>
void NoiseAnalysis<NoiseModelT>::visitExternalCall(
    CallOpInterface call, ArrayRef<const LatticeType*> argumentLattices,
    ArrayRef<LatticeType*> resultLattices) {
  ::mlir::heir::visitExternalCall<NoiseState, LatticeType>(
      call, argumentLattices, resultLattices,
      [this](AnalysisState* state, ChangeResult changed) {
        this->propagateIfChanged(state, changed);
      });
}

template <typename NoiseModelT>
LogicalResult NoiseAnalysis<NoiseModelT>::visitOperation(
    Operation* op, ArrayRef<const LatticeType*> operands,
    ArrayRef<LatticeType*> results) {
  LLVM_DEBUG(llvm::dbgs() << "NoiseAnalysis: Visiting op " << *op << "\n");
  auto getLocalParam = [&](Value value) {
    auto level = getLevelFromMgmtAttr(value).getInt();
    auto dimension = getDimensionFromMgmtAttr(value);
    return LocalParamType(&schemeParam, level, dimension);
  };

  auto propagate = [&](Value value, NoiseState noise) {
    LLVM_DEBUG(llvm::dbgs() << "Propagating "
                            << doubleToString2Prec(noiseModel.toLogBound(
                                   getLocalParam(value), noise))
                            << " to " << value << "\n");
    LatticeType* lattice = this->getLatticeElement(value);
    auto changeResult = lattice->join(noise);
    this->propagateIfChanged(lattice, changeResult);
  };

  auto getOperandNoises = [&](Operation* op,
                              SmallVectorImpl<NoiseState>& noises) {
    SmallVector<OpOperand*> secretOperands;
    SmallVector<OpOperand*> nonSecretOperands;
    this->getSecretOperands(op, secretOperands);
    this->getPlaintextOperands(op, nonSecretOperands);

    for (auto* operand : secretOperands) {
      noises.push_back(this->getLatticeElement(operand->get())->getValue());
    }
    for (auto* operand : nonSecretOperands) {
      (void)operand;
      // at least one operand is secret
      auto localParam = getLocalParam(secretOperands[0]->get());
      noises.push_back(noiseModel.evalConstant(localParam));
    }
  };

  auto res =
      llvm::TypeSwitch<Operation&, LogicalResult>(*op)
          .template Case<secret::RevealOp, secret::ConcealOp>([&](auto op) {
            for (auto result : results) {
              setToEntryState(result);
            }
            return success();
          })
          .template Case<secret::GenericOp>([&](auto genericOp) {
            Block* body = genericOp.getBody();
            for (Value& arg : body->getArguments()) {
              auto localParam = getLocalParam(arg);
              NoiseState encrypted = noiseModel.evalEncrypt(localParam);
              propagate(arg, encrypted);
            }
            return success();
          })
          .template Case<arith::MulIOp>([&](auto mulOp) {
            SmallVector<OpResult> secretResults;
            this->getSecretResults(mulOp, secretResults);
            if (secretResults.empty()) {
              return success();
            }

            SmallVector<NoiseState, 2> operandNoises;
            getOperandNoises(mulOp, operandNoises);

            auto localParam = getLocalParam(mulOp.getResult());
            NoiseState mult = noiseModel.evalMul(localParam, operandNoises[0],
                                                 operandNoises[1]);
            propagate(mulOp.getResult(), mult);
            return success();
          })
          .template Case<arith::AddIOp, arith::SubIOp>([&](auto addOp) {
            SmallVector<OpResult> secretResults;
            this->getSecretResults(addOp, secretResults);
            if (secretResults.empty()) {
              return success();
            }

            SmallVector<NoiseState, 2> operandNoises;
            getOperandNoises(addOp, operandNoises);
            NoiseState add =
                noiseModel.evalAdd(operandNoises[0], operandNoises[1]);
            propagate(addOp.getResult(), add);
            return success();
          })
          .template Case<tensor_ext::RotateOp>([&](auto rotateOp) {
            auto localParam = getLocalParam(rotateOp.getOperand(0));
            NoiseState rotate =
                noiseModel.evalRelinearize(localParam, operands[0]->getValue());
            propagate(rotateOp.getResult(), rotate);
            return success();
          })
          .template Case<mgmt::AdjustScaleOp>([&](auto adjustScaleOp) {
            if constexpr (is_bfv_noise_model<NoiseModelT>::value) {
              adjustScaleOp->emitError(
                  "Unsupported operation for noise analysis encountered.");
              return failure();
            } else {
              auto localParam = getLocalParam(adjustScaleOp.getInput());
              NoiseState someFactor = noiseModel.evalConstant(localParam);
              NoiseState mulConst = noiseModel.evalMul(
                  localParam, operands[0]->getValue(), someFactor);
              propagate(adjustScaleOp.getResult(), mulConst);
              return success();
            }
          })
          .template Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
            if constexpr (is_bfv_noise_model<NoiseModelT>::value) {
              modReduceOp->emitWarning("ModReduceOp encountered in BFV");
              propagate(modReduceOp.getResult(), operands[0]->getValue());
              return success();
            } else {
              auto localParam = getLocalParam(modReduceOp.getInput());
              NoiseState modReduce =
                  noiseModel.evalModReduce(localParam, operands[0]->getValue());
              propagate(modReduceOp.getResult(), modReduce);
              return success();
            }
          })
          .template Case<mgmt::LevelReduceOp>([&](auto levelReduceOp) {
            propagate(levelReduceOp.getResult(), operands[0]->getValue());
            return success();
          })
          .template Case<mgmt::RelinearizeOp>([&](auto relinearizeOp) {
            auto localParam = getLocalParam(relinearizeOp.getInput());
            NoiseState relinearize =
                noiseModel.evalRelinearize(localParam, operands[0]->getValue());
            propagate(relinearizeOp.getResult(), relinearize);
            return success();
          })
          .Default([&](auto& op) {
            SmallVector<OpResult> secretResults;
            this->getSecretResults(&op, secretResults);
            if (secretResults.empty()) {
              return success();
            }

            if (!mlir::isa<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp,
                           arith::ExtFOp, mgmt::InitOp, tensor::ExtractSliceOp,
                           tensor::InsertSliceOp>(op)) {
              op.emitError()
                  << "Unsupported operation for noise analysis encountered.";
            }

            SmallVector<OpOperand*> secretOperands;
            this->getSecretOperands(&op, secretOperands);
            if (secretOperands.empty()) {
              return success();
            }

            NoiseState first;
            for (auto* operand : secretOperands) {
              auto& noise = this->getLatticeElement(operand->get())->getValue();
              if (!noise.isInitialized()) {
                return success();
              }
              first = noise;
              break;
            }

            for (auto result : secretResults) {
              propagate(result, first);
            }
            return success();
          });
  return res;
}

}  // namespace heir
}  // namespace mlir

#ifdef HEIR_NOISE_ANALYSIS_UNDEF_DEBUG_TYPE
#undef DEBUG_TYPE
#undef HEIR_NOISE_ANALYSIS_UNDEF_DEBUG_TYPE
#endif

#endif  // LIB_ANALYSIS_NOISEANALYSIS_NOISEANALYSIS_H_
