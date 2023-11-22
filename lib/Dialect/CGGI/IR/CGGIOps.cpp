#ifndef LIB_DIALECT_CGGI_IR_CGGIOPS_CPP_
#define LIB_DIALECT_CGGI_IR_CGGIOPS_CPP_

#include "include/Dialect/CGGI/IR/CGGIOps.h"

#include "include/Analysis/NoisePropagation/Variance.h"
#include "include/Dialect/CGGI/IR/CGGIAttributes.h"
#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "include/Interfaces/NoiseInterfaces.h"

namespace mlir {
namespace heir {
namespace cggi {

unsigned maxPerDigitDecompositionError(unsigned baseLog, unsigned numLevels,
                                       unsigned ctBitWidth) {
  // FIXME: this needs verification; I struggled to parse what was said in the
  // CGGI paper, as well as the original DM paper, so I relied on my own
  // analysis in https://jeremykun.com/2022/08/29/key-switching-in-lwe/
  // It aligns roughly with the error analysis in Theorem 4.1 of
  // https://eprint.iacr.org/2018/421, but using a different perspective
  // on the "precision" parameter t in that paper.

  // maxLevels is the number L such that B^L = lwe_cmod
  // a.k.a., L * log2(B) = cmod_bitwidth
  // This should be an exact division, since the LWE cmod is always supposed to
  // be a power of two.
  unsigned maxLevels = ctBitWidth / baseLog;
  unsigned lowestLevel = maxLevels - numLevels;
  // Regardless of whether the approximation is signed or not, the max error you
  // can get per digit is to be off by B-1.
  unsigned approximationPerDigitMaxError = (1 << baseLog) - 1;
  return (unsigned)pow(approximationPerDigitMaxError, lowestLevel - 1);
}

/// This function represents one noise model for the output of bootstrap in the
/// simplest CGGI implementation. It is an upper bound estimate of the variance
/// of a ciphertext post bootstrap (including the key switch op). Follows the
/// formula in https://eprint.iacr.org/2018/421, Theorem 6.3.
///
/// Notes:
///
/// In the paper, the key-switching key gadget is binary. Here it has an
/// arbitrary base and number of levels.
///
/// Signed decompositions are used for the gadgets, leading to a multiplicative
/// factor of two difference between the quality "beta" (max digit size) of the
/// gadget and the chosen parameter 2**base_log.
int64_t bootstrapOutputNoise(CGGIParamsAttr attr,
                             lwe::LWEParamsAttr lweParams) {
  lwe::RLWEParamsAttr rlweParams = attr.getRlweParams();
  unsigned bskNoiseVariance = attr.getBskNoiseVariance();
  unsigned kskNoiseVariance = attr.getKskNoiseVariance();

  // Mirroring the notation in https://eprint.iacr.org/2018/421, Theorem 6.3.
  unsigned logq = lweParams.getCmod().getValue().getBitWidth();
  unsigned n = lweParams.getDimension();
  unsigned k = rlweParams.getDimension();
  unsigned N = rlweParams.getPolyDegree();
  unsigned l = attr.getBskGadgetNumLevels();
  // Beta is the max absolute value of a digit of the signed decomposition
  unsigned beta = (1 << attr.getBskGadgetBaseLog()) / 2;

  // Epsilon is the max per-digit error of the approximation introduced by
  // having fewer levels in the gadget key.
  // FIXME: this needs verification. I think it's the same sort of error as the
  // key switching key sampleApproxError below.
  unsigned epsilon = maxPerDigitDecompositionError(
      attr.getBskGadgetBaseLog(), attr.getBskGadgetNumLevels(), logq);
  unsigned externalProductTerm =
      (n * (k + 1) * l * N * beta * beta * bskNoiseVariance +
       n * (1 + k * N) * epsilon * epsilon);

  // largestDigit depends on a signed decomposition.
  unsigned largestDigit = (1 << attr.getKskGadgetBaseLog()) / 2;
  unsigned kskSampleApproxError = maxPerDigitDecompositionError(
      attr.getKskGadgetBaseLog(), attr.getKskGadgetNumLevels(), logq);
  unsigned keySwitchingTerm =
      (attr.getKskGadgetNumLevels() * largestDigit * kskNoiseVariance +
       n * kskSampleApproxError);
  return externalProductTerm + keySwitchingTerm;
}

void handleSingleResultOp(Operation *op, Value ctValue,
                          SetNoiseFn setValueNoise) {
  auto lweParams =
      cast<lwe::LWECiphertextType>(ctValue.getType()).getLweParams();
  if (!lweParams) {
    op->emitOpError() << "lwe_params must be set on the input values to run "
                         "noise propagation.";
    return;
  }

  auto attrs = op->getAttrDictionary();
  if (!attrs.contains("cggi_params")) {
    op->emitOpError() << "cggi_params must be set to run noise propagation.";
    return;
  }
  auto cggiParams = llvm::cast<CGGIParamsAttr>(attrs.get("cggi_params"));
  setValueNoise(op->getResult(0),
                Variance(bootstrapOutputNoise(cggiParams, lweParams)));
}

void AndOp::inferResultNoise(llvm::ArrayRef<Variance> argNoises,
                             SetNoiseFn setValueNoise) {
  return handleSingleResultOp(getOperation(), getLhs(), setValueNoise);
}

void OrOp::inferResultNoise(llvm::ArrayRef<Variance> argNoises,
                            SetNoiseFn setValueNoise) {
  return handleSingleResultOp(getOperation(), getLhs(), setValueNoise);
}

void XorOp::inferResultNoise(llvm::ArrayRef<Variance> argNoises,
                             SetNoiseFn setValueNoise) {
  return handleSingleResultOp(getOperation(), getLhs(), setValueNoise);
}

void Lut3Op::inferResultNoise(llvm::ArrayRef<Variance> argNoises,
                              SetNoiseFn setValueNoise) {
  return handleSingleResultOp(getOperation(), getA(), setValueNoise);
}

void Lut2Op::inferResultNoise(llvm::ArrayRef<Variance> argNoises,
                              SetNoiseFn setValueNoise) {
  return handleSingleResultOp(getOperation(), getA(), setValueNoise);
}

void NotOp::inferResultNoise(llvm::ArrayRef<Variance> argNoises,
                             SetNoiseFn setValueNoise) {
  // This one doesn't use bootstrap, no error change
  setValueNoise(getInput(), argNoises[0]);
}

}  // namespace cggi
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CGGI_IR_CGGIOPS_CPP_
