#ifndef LIB_TRANSFORMS_FOLDPLAINTEXTMASKS_FOLDPLAINTEXTMASKS_TD_
#define LIB_TRANSFORMS_FOLDPLAINTEXTMASKS_FOLDPLAINTEXTMASKS_TD_

include "mlir/Pass/PassBase.td"

def FoldPlaintextMasks : Pass<"fold-plaintext-masks"> {
  let summary = "Apply folding rules for tensor masking operations";
  let description = [{
  This pass applies a set of folding rules to optimize ciphertext-plaintext
  masking operations that occur on ciphertext-semantic tensors.

  This typically corresponds to applications of `arith.muli` or `arith.mulf`
  where one operand is a constant tensor consisting of 1's and 0's. Because
  these apply even when the operand tensors are not specifically
  ciphertext-semantic tensors, this pass can run on any IR and it will
  still produce semantically correct results.

  (* example filepath=tests/Transforms/fold_plaintext_masks/doctest.mlir *)
  }];
  let dependentDialects = [

  ];
}

#endif  // LIB_TRANSFORMS_FOLDPLAINTEXTMASKS_FOLDPLAINTEXTMASKS_TD_
