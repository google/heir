#ifndef LIB_TRANSFORMS_PROPAGATEPADDING_PROPAGATEPADDING_TD_
#define LIB_TRANSFORMS_PROPAGATEPADDING_PROPAGATEPADDING_TD_

include "mlir/Pass/PassBase.td"

def PropagatePadding : Pass<"propagate-padding"> {
  let summary = "Propagate logical-vs-padded shape information from tensor.pad ops";
  let description = [{
    Walks each function in program order and derives, for every value it can
    soundly reason about, a `#tensor_ext.padding` attribute recording the
    value's logical shape, its padded shape, and whether the padding region
    is exactly zero.

    Information is seeded at `tensor.pad` ops (static, low-padding-free,
    zero-valued pads) and transferred through ops implementing
    `PaddingSemanticsOpInterface` (matmuls, elementwise arithmetic,
    transpose, broadcast, reduce, ...). An op with no implementation, or an
    implementation that cannot make a sound claim, simply stops propagation:
    downstream consumers then see no attribute and must behave as if the
    value were unpadded, which matches the semantics of circuits that never
    contained padding.

    The derived attribute is stamped on the defining op as
    `tensor_ext.padding` and verified against the op's result type: the
    padded shape must equal the actual tensor shape and the logical shape
    must be elementwise no larger. A violation is a hard error.

    Because discardable attributes do not survive pattern rewrites that
    recreate ops, this pass should run immediately before the passes that
    consume the attribute (e.g. softmax lowerings), with no canonicalization
    in between.
  }];
  let dependentDialects = [
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::heir::tensor_ext::TensorExtDialect"
  ];
}

#endif  // LIB_TRANSFORMS_PROPAGATEPADDING_PROPAGATEPADDING_TD_
