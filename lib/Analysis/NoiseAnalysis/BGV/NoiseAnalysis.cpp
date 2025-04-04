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
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
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
  // At an entry point, we have no information about the noise.
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
  auto getLocalParam = [&](Value value) {
    auto level = getLevelFromMgmtAttr(value);
    auto dimension = getDimensionFromMgmtAttr(value);
    return LocalParamType(&schemeParam, level, dimension);
  };

  auto propagate = [&](Value value, NoiseState noise) {
    LLVM_DEBUG(llvm::dbgs()
               << "Propagating "
               << NoiseModel::toLogBoundString(getLocalParam(value), noise)
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
      noises.push_back(NoiseModel::evalConstant(localParam));
    }
  };

  auto res =
      llvm::TypeSwitch<Operation &, LogicalResult>(*op)
          .Case<secret::GenericOp>([&](auto genericOp) {
            Block *body = genericOp.getBody();
            for (Value &arg : body->getArguments()) {
              auto localParam = getLocalParam(arg);
              NoiseState encrypted = NoiseModel::evalEncrypt(localParam);
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
            NoiseState mult = NoiseModel::evalMul(localParam, operandNoises[0],
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
                NoiseModel::evalAdd(operandNoises[0], operandNoises[1]);
            propagate(addOp.getResult(), add);
            return success();
          })
          .template Case<tensor_ext::RotateOp>([&](auto rotateOp) {
            // implicitly assumed secret
            auto localParam = getLocalParam(rotateOp.getOperand(0));

            // assume relinearize immediately after rotate
            // when we support hoisting relinearize, we need to change
            // this
            NoiseState rotate = NoiseModel::evalRelinearize(
                localParam, operands[0]->getValue());
            propagate(rotateOp.getResult(), rotate);
            return success();
          })
          // NOTE: special case for ExtractOp... it is a mulconst+rotate
          // if not annotated with slot_extract
          // TODO(#1174): decide packing earlier in the pipeline instead of
          // annotation
          .template Case<tensor::ExtractOp>([&](auto extractOp) {
            auto localParam = getLocalParam(extractOp.getOperand(0));

            // extract = mul_plain 1 + rotate
            // although the cleartext is 1, when encoded (i.e. CRT
            // packing), the value multiplied to the ciphertext is not 1,
            // If we can know the encoded value, we can bound it more precisely.
            NoiseState one = NoiseModel::evalConstant(localParam);
            NoiseState extract =
                NoiseModel::evalMul(localParam, operands[0]->getValue(), one);
            // assume relinearize immediately after rotate
            // when we support hoisting relinearize, we need to change
            // this
            NoiseState rotate =
                NoiseModel::evalRelinearize(localParam, extract);
            propagate(extractOp.getResult(), extract);
            return success();
          })
          .template Case<mgmt::AdjustScaleOp>([&](auto adjustScaleOp) {
            auto localParam = getLocalParam(adjustScaleOp.getInput());

            // adjust scale materializes to a mulconst
            NoiseState someFactor = NoiseModel::evalConstant(localParam);
            NoiseState mulConst = NoiseModel::evalMul(
                localParam, operands[0]->getValue(), someFactor);
            propagate(adjustScaleOp.getResult(), mulConst);
            return success();
          })
          .template Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
            auto localParam = getLocalParam(modReduceOp.getInput());

            NoiseState modReduce =
                NoiseModel::evalModReduce(localParam, operands[0]->getValue());
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

            NoiseState relinearize = NoiseModel::evalRelinearize(
                localParam, operands[0]->getValue());
            propagate(relinearizeOp.getResult(), relinearize);
            return success();
          })
          .Default([&](auto &op) {
            if (!mlir::isa<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp,
                           arith::ExtFOp, mgmt::InitOp>(op)) {
              op.emitError()
                  << "Unsupported operation for noise analysis encountered.";
            }

            // condition on result secretness
            SmallVector<OpResult> secretResults;
            this->getSecretResults(&op, secretResults);
            if (secretResults.empty()) {
              return success();
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
template class NoiseAnalysis<bgv::NoiseByBoundCoeffAverageCaseModel>;
template class NoiseAnalysis<bgv::NoiseByBoundCoeffWorstCaseModel>;

// for mono bounds
template class NoiseAnalysis<bgv::NoiseCanEmbModel>;

// for by variance
template class NoiseAnalysis<bgv::NoiseByVarianceCoeffModel>;

}  // namespace heir
}  // namespace mlir
