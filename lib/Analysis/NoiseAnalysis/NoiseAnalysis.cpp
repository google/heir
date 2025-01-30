#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/NoiseAnalysis/BGV/Noise.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project

#define DEBUG_TYPE "NoiseAnalysis"

namespace mlir {
namespace heir {

template <typename Noise>
LogicalResult NoiseAnalysis<Noise>::visitOperation(
    Operation *op, ArrayRef<const NoiseLattice<Noise> *> operands,
    ArrayRef<NoiseLattice<Noise> *> results) {
  auto getLocalParam = [&](Value value) {
    auto level = getLevelFromMgmtAttr(value);
    auto dimension = getDimensionFromMgmtAttr(value);
    return LocalParam(&schemeParam, level, dimension);
  };

  auto propagate = [&](Value value, Noise noise) {
    auto localParam = getLocalParam(value);

    LLVM_DEBUG(llvm::dbgs() << "Propagating " << noise.toBound(localParam)
                            << " to " << value << "\n");
    NoiseLattice<Noise> *lattice = this->getLatticeElement(value);
    auto changeResult = lattice->join(noise);
    this->propagateIfChanged(lattice, changeResult);
  };

  auto getOperandNoises = [&](Operation *op, SmallVectorImpl<Noise> &noises) {
    SmallVector<OpOperand *> secretOperands;
    SmallVector<OpOperand *> nonSecretOperands;
    this->getSecretOperands(op, secretOperands);
    this->getNonSecretOperands(op, nonSecretOperands);

    for (auto *operand : secretOperands) {
      noises.push_back(this->getLatticeElement(operand->get())->getValue());
    }
    for (auto *operand : nonSecretOperands) {
      // at least one operand is secret
      auto localParam = getLocalParam(secretOperands[0]->get());
      noises.push_back(Noise::evalConstant(localParam));
    }
  };

  auto res =
      llvm::TypeSwitch<Operation &, LogicalResult>(*op)
          .Case<secret::GenericOp>([&](auto genericOp) {
            Block *body = genericOp.getBody();
            for (Value &arg : body->getArguments()) {
              auto localParam = getLocalParam(arg);
              Noise encrypted = Noise::evalEncryptPk(localParam);
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

            SmallVector<Noise, 2> operandNoises;
            getOperandNoises(mulOp, operandNoises);

            auto localParam = getLocalParam(mulOp.getResult());
            // TODO: handle mixed degree op
            Noise mult = Noise::evalMultNoRelin(localParam, operandNoises[0],
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

            SmallVector<Noise, 2> operandNoises;
            getOperandNoises(addOp, operandNoises);
            // TODO: Handle mixed degree op
            Noise add = Noise::evalAdd(operandNoises[0], operandNoises[1]);
            propagate(addOp.getResult(), add);
            return success();
          })
          .template Case<tensor_ext::RotateOp>([&](auto rotateOp) {
            // implicitly assumed secret
            auto localParam = getLocalParam(rotateOp.getOperand(0));

            Noise rotate =
                Noise::evalRotate(localParam, operands[0]->getValue());
            propagate(rotateOp.getResult(), rotate);
            return success();
          })
          .template Case<tensor::ExtractOp>([&](auto extractOp) {
            auto localParam = getLocalParam(extractOp.getOperand(0));

            // extract = mul + rotate
            Noise constant = Noise::evalConstant(localParam);
            Noise extract = Noise::evalMultNoRelin(
                localParam, operands[0]->getValue(), constant);
            propagate(extractOp.getResult(), extract);
            return success();
          })
          .template Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
            auto localParam = getLocalParam(modReduceOp.getInput());

            Noise modReduce =
                Noise::evalModReduce(localParam, operands[0]->getValue());
            propagate(modReduceOp.getResult(), modReduce);
            return success();
          })
          .template Case<mgmt::RelinearizeOp>([&](auto relinearizeOp) {
            auto localParam = getLocalParam(relinearizeOp.getInput());

            Noise relinearize =
                Noise::evalRelinearize(localParam, operands[0]->getValue());
            propagate(relinearizeOp.getResult(), relinearize);
            return success();
          })
          .Default([&](auto &op) { return success(); });
  return res;
}

// instantiation
template class NoiseAnalysis<bgv::Noise<true>>;
template class NoiseAnalysis<bgv::Noise<false>>;

}  // namespace heir
}  // namespace mlir
