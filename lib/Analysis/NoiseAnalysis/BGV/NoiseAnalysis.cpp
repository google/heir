#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"

#include <functional>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByVarianceCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseCanEmbModel.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project

#define DEBUG_TYPE "NoiseAnalysis"

namespace mlir {
namespace heir {

// explicit specialization of NoiseAnalysis for NoiseByBoundCoeffModel
template <typename NoiseModel>
void NoiseAnalysis<NoiseModel>::setToEntryState(LatticeType *lattice) {
  if (isa<secret::SecretType>(lattice->getAnchor().getType())) {
    Value value = lattice->getAnchor();
    auto localParam = LocalParamType(&schemeParam, getLevelFromMgmtAttr(value),
                                     getDimensionFromMgmtAttr(value));
    NoiseState encrypted = noiseModel.evalEncrypt(localParam);
    this->propagateIfChanged(lattice, lattice->join(encrypted));
    return;
  }

  this->propagateIfChanged(lattice, lattice->join(NoiseState::uninitialized()));
}

// explicit specialization of NoiseAnalysis for NoiseByBoundCoeffModel
template <typename NoiseModel>
void NoiseAnalysis<NoiseModel>::visitExternalCall(
    CallOpInterface call, ArrayRef<const LatticeType *> argumentLattices,
    ArrayRef<LatticeType *> resultLattices) {
  auto callback =
      std::bind(&NoiseAnalysis<NoiseModel>::propagateIfChangedWrapper, this,
                std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<NoiseState, LatticeType>(
      call, argumentLattices, resultLattices, callback);
}

// explicit specialization of NoiseAnalysis for NoiseByBoundCoeffModel
template <typename NoiseModel>
LogicalResult NoiseAnalysis<NoiseModel>::visitOperation(
    Operation *op, ArrayRef<const LatticeType *> operands,
    ArrayRef<LatticeType *> results) {
  LLVM_DEBUG(llvm::dbgs() << "NoiseAnalysis: Visiting op " << *op << "\n");
  auto getLocalParam = [&](Value value) {
    auto level = getLevelFromMgmtAttr(value);
    auto dimension = getDimensionFromMgmtAttr(value);
    return LocalParamType(&schemeParam, level, dimension);
  };

  auto propagate = [&](Value value, NoiseState noise) {
    LLVM_DEBUG(llvm::dbgs() << "Propagating "
                            << doubleToString2Prec(noiseModel.toLogBound(
                                   getLocalParam(value), noise))
                            << " to " << value << "\n");
    LatticeType *lattice = this->getLatticeElement(value);
    auto changeResult = lattice->join(noise);
    this->propagateIfChanged(lattice, changeResult);
  };

  auto getOperandNoises = [&](Operation *op,
                              SmallVectorImpl<NoiseState> &noises) {
    SmallVector<OpOperand *> secretOperands;
    SmallVector<OpOperand *> nonSecretOperands;
    this->getSecretOperands(op, secretOperands);
    this->getNonSecretOperands(op, nonSecretOperands);

    for (auto *operand : secretOperands) {
      noises.push_back(this->getLatticeElement(operand->get())->getValue());
    }
    for (auto *operand : nonSecretOperands) {
      (void)operand;
      // at least one operand is secret
      auto localParam = getLocalParam(secretOperands[0]->get());
      noises.push_back(noiseModel.evalConstant(localParam));
    }
  };

  auto res =
      llvm::TypeSwitch<Operation &, LogicalResult>(*op)
          .template Case<secret::RevealOp, secret::ConcealOp>([&](auto op) {
            // Reveal outputs are not secret, so no noise. Conceal outputs are
            // a fresh encryption. Both are handled properly by setToEntryState
            // based on the type of the result.
            for (auto result : results) {
              setToEntryState(result);
            }
            return success();
          })
          .template Case<secret::GenericOp>([&](auto genericOp) {
            Block *body = genericOp.getBody();
            for (Value &arg : body->getArguments()) {
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
            // implicitly assumed secret
            auto localParam = getLocalParam(rotateOp.getOperand(0));

            // assume relinearize immediately after rotate
            // when we support hoisting relinearize, we need to change
            // this
            NoiseState rotate =
                noiseModel.evalRelinearize(localParam, operands[0]->getValue());
            propagate(rotateOp.getResult(), rotate);
            return success();
          })
          .template Case<mgmt::AdjustScaleOp>([&](auto adjustScaleOp) {
            auto localParam = getLocalParam(adjustScaleOp.getInput());

            // adjust scale materializes to a mulconst
            NoiseState someFactor = noiseModel.evalConstant(localParam);
            NoiseState mulConst = noiseModel.evalMul(
                localParam, operands[0]->getValue(), someFactor);
            propagate(adjustScaleOp.getResult(), mulConst);
            return success();
          })
          .template Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
            auto localParam = getLocalParam(modReduceOp.getInput());

            NoiseState modReduce =
                noiseModel.evalModReduce(localParam, operands[0]->getValue());
            propagate(modReduceOp.getResult(), modReduce);
            return success();
          })
          .template Case<mgmt::LevelReduceOp>([&](auto levelReduceOp) {
            // preserve noise
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
          .Default([&](auto &op) {
            // condition on result secretness
            SmallVector<OpResult> secretResults;
            this->getSecretResults(&op, secretResults);
            if (secretResults.empty()) {
              return success();
            }

            if (!mlir::isa<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp,
                           arith::ExtFOp, mgmt::InitOp>(op)) {
              op.emitError()
                  << "Unsupported operation for noise analysis encountered.";
            }

            SmallVector<OpOperand *> secretOperands;
            this->getSecretOperands(&op, secretOperands);
            if (secretOperands.empty()) {
              return success();
            }

            // inherit noise from the first secret operand
            NoiseState first;
            for (auto *operand : secretOperands) {
              auto &noise = this->getLatticeElement(operand->get())->getValue();
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

// template instantiation
template class NoiseAnalysis<bgv::NoiseByBoundCoeffModel>;

// for mono bounds
template class NoiseAnalysis<bgv::NoiseCanEmbModel>;

// for by variance
template class NoiseAnalysis<bgv::NoiseByVarianceCoeffModel>;

}  // namespace heir
}  // namespace mlir
