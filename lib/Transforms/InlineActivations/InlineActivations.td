#ifndef LIB_TRANSFORMS_INLINEACTIVATIONS_INLINEACTIVATIONS_TD_
#define LIB_TRANSFORMS_INLINEACTIVATIONS_INLINEACTIVATIONS_TD_

include "mlir/Pass/PassBase.td"

def InlineActivations : Pass<"inline-activations"> {
  let summary = "Inline activation functions";
  let description = [{
   This pass inlines activation functions into the IR.

   Conversion paths from PyTorch to linalg use PyTorch/XLA utilities that
   convert the torch activation functions into separate functions. This pass
   inlines these functions into the IR. The candidate activations are identified
   with an allowlist of function names and are otherwise not checked for
   correctness. See
   https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.

   (* example filepath=tests/Transforms/inline_activations/relu.mlir *)
  }];
}

#endif  // LIB_TRANSFORMS_INLINEACTIVATIONS_INLINEACTIVATIONS_TD_
