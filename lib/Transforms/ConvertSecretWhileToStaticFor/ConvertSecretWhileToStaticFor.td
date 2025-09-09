#ifndef LIB_TRANSFORMS_CONVERTSECRETWHILETOSTATICFOR_CONVERTSECRETWHILETOSTATICFOR_TD_
#define LIB_TRANSFORMS_CONVERTSECRETWHILETOSTATICFOR_CONVERTSECRETWHILETOSTATICFOR_TD_

include "mlir/Pass/PassBase.td"

def ConvertSecretWhileToStaticFor : Pass<"convert-secret-while-to-static-for"> {
  let summary = "Convert secret scf.while ops to affine.for ops that have constant bounds.";
  let description = [{
    Convert scf.while with a secret condition to affine.for with constant
    bounds. It replaces the scf.condition operation found in the scf.while loop
    with an scf.if operation that conditionally executes operations in the while
    operation's body and yields values.

    A "max_iter" attribute should be specified as part of the secret-dependent
    scf.while operation to successfully transform to a secret-independent
    affine.for operation. This attribute determines the maximum number of
    iterations for the new affine.for operation.

    Note: Running this pass alone does not result in a data-oblivious program;
    we have to run the `--convert-if-to-select` pass to the resulting program
    to convert the secret-dependent If-operation to a Select-operation.

    (* example filepath=tests/Transforms/convert_secret_while_to_static_for/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif  // LIB_TRANSFORMS_CONVERTSECRETWHILETOSTATICFOR_CONVERTSECRETWHILETOSTATICFOR_TD_
