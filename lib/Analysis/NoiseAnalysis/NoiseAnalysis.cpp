#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"

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

LogicalResult NoiseAnalysis::visitOperation(
    Operation *op, ArrayRef<const NoiseLattice *> operands,
    ArrayRef<NoiseLattice *> results) {
  auto getLocalParam = [&](Value value) -> std::optional<LocalParam> {
    return LocalParam(&schemeParam, 0, 2);
  };

  auto propagate = [&](Value value, Noise noise) {
    auto localParam = getLocalParam(value).value();

    // LLVM_DEBUG(llvm::dbgs() << "Propagating " << noise.toBound(localParam)
    //                         << " to " << value << "\n");
    NoiseLattice *lattice = getLatticeElement(value);
    auto changeResult = lattice->join(noise);
    propagateIfChanged(lattice, changeResult);
  };

  auto res =
      llvm::TypeSwitch<Operation &, LogicalResult>(*op)
          .Case<secret::GenericOp>([&](auto genericOp) {
            Block *body = genericOp.getBody();
            for (Value &arg : body->getArguments()) {
              auto localParamOpt = getLocalParam(arg);
              if (!localParamOpt.has_value()) {
                return success();
              }

              auto localParam = *localParamOpt;

              Noise encrypted = Noise::evalEncryptPk(localParam);
              propagate(arg, encrypted);
            }
            return success();
          })
          .Case<arith::ConstantOp>([&](auto constantOp) {
            auto localParamOpt = getLocalParam(constantOp.getResult());
            if (!localParamOpt.has_value()) {
              return success();
            }

            auto localParam = *localParamOpt;
            Noise constant = Noise::evalConstant(localParam);
            propagate(constantOp.getResult(), constant);
            return success();
          })
          .Case<arith::MulIOp>([&](auto mulOp) {
            auto localParamOpt = getLocalParam(mulOp.getResult());
            if (!localParamOpt.has_value()) {
              return success();
            }

            auto localParam = *localParamOpt;
            Noise mult = Noise::evalMultNoRelin(
                localParam, operands[0]->getValue(), operands[1]->getValue());
            propagate(mulOp.getResult(), mult);
            return success();
          })
          .Case<arith::AddIOp, arith::SubIOp>([&](auto addOp) {
            Noise add = Noise::evalAdd(operands[0]->getValue(),
                                       operands[1]->getValue());
            propagate(addOp.getResult(), add);
            return success();
          })
          .Case<tensor_ext::RotateOp>([&](auto rotateOp) {
            auto localParamOpt = getLocalParam(rotateOp.getOperand(0));
            if (!localParamOpt.has_value()) {
              return success();
            }

            auto localParam = *localParamOpt;
            Noise rotate =
                Noise::evalRotate(localParam, operands[0]->getValue());
            propagate(rotateOp.getResult(), rotate);
            return success();
          })
          .Case<tensor::ExtractOp>([&](auto extractOp) {
            auto localParamOpt = getLocalParam(extractOp.getOperand(0));
            if (!localParamOpt.has_value()) {
              return success();
            }

            auto localParam = *localParamOpt;
            // extract = mul + rotate
            Noise constant = Noise::evalConstant(localParam);
            Noise extract = Noise::evalMultNoRelin(
                localParam, operands[0]->getValue(), constant);
            propagate(extractOp.getResult(), extract);
            return success();
          })
          .Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
            auto localParamOpt = getLocalParam(modReduceOp.getInput());
            if (!localParamOpt.has_value()) {
              return success();
            }

            auto localParam = *localParamOpt;
            Noise modReduce =
                Noise::evalModReduce(localParam, operands[0]->getValue());
            propagate(modReduceOp.getResult(), modReduce);
            return success();
          })
          .Case<mgmt::RelinearizeOp>([&](auto relinearizeOp) {
            auto localParamOpt = getLocalParam(relinearizeOp.getInput());
            if (!localParamOpt.has_value()) {
              return success();
            }

            auto localParam = *localParamOpt;
            Noise relinearize =
                Noise::evalRelinearize(localParam, operands[0]->getValue());
            propagate(relinearizeOp.getResult(), relinearize);
            return success();
          })
          .Default([&](auto &op) { return success(); });
  return res;
}

}  // namespace heir
}  // namespace mlir
