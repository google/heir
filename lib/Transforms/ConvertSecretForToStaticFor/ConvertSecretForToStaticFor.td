#ifndef LIB_TRANSFORMS_CONVERTSECRETFORTOSTATICFOR_CONVERTSECRETFORTOSTATICFOR_TD_
#define LIB_TRANSFORMS_CONVERTSECRETFORTOSTATICFOR_CONVERTSECRETFORTOSTATICFOR_TD_

include "mlir/Pass/PassBase.td"

def ConvertSecretForToStaticFor : Pass<"convert-secret-for-to-static-for"> {
  let summary = "Convert secret scf.for ops to affine.for ops with constant bounds.";
  let description = [{
    Conversion for For-operation that evaluate secret bound(s) to alternative
    affine For-operation with constant bound(s).

    It replaces data-dependent bounds with an If-operation to check the bounds,
    and conditionally execute and yield values from the For-operation's body.
    Note: Running this pass alone does not result in a data-oblivious program;
    we have to run the `--convert-if-to-select` pass to the resulting program
    to convert the secret-dependent If-operation to a Select-operation.

    (* example filepath=tests/Transforms/convert_secret_for_to_static_for/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect"
  ];
  let options = [
    Option<"convertAllScfFor", "convert-all-scf-for", "bool", /*default=*/"true",
           "If true, convert all scf.for ops to affine.for, not just those with secret bounds.">,
  ];
}

#endif  // LIB_TRANSFORMS_CONVERTSECRETFORTOSTATICFOR_CONVERTSECRETFORTOSTATICFOR_TD_
