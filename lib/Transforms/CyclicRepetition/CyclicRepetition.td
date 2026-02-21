#ifndef LIB_TRANSFORMS_CYCLICREPETITION_CYCLICREPETITION_TD_
#define LIB_TRANSFORMS_CYCLICREPETITION_CYCLICREPETITION_TD_

include "mlir/Pass/PassBase.td"

def CyclicRepetition : Pass<"cyclic-repetition", "ModuleOp"> {
  let summary = "Cyclically repeat inputs to lwe.rlwe_encode for slot count mismatches.";
  let description = [{
    This pass looks for `lwe.rlwe_encode` operations and cyclically repeats
    the input tensor to match the actual slot count of the ring, if the
    actual slot count is larger than the requested slot count.

    In particular, parameter selection passes in HEIR may select a larger
    slot count than is configured via the `ciphertext-degree` pipeline input.
    In that case, a kernel like Halevi-Shoup matvec will not work properly
    without cyclic repetition of the packed plaintexts and ciphertexts to fill
    the available slots.
  }];
  let dependentDialects = ["mlir::heir::lwe::LWEDialect", "mlir::tensor::TensorDialect", "mlir::arith::ArithDialect"];
}

#endif // LIB_TRANSFORMS_CYCLICREPETITION_CYCLICREPETITION_TD_
