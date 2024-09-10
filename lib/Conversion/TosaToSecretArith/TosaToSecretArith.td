#ifndef LIB_CONVERSION_TOSATOSECRETARITH_TOSATOSECRETARITH_TD_
#define LIB_CONVERSION_TOSATOSECRETARITH_TOSATOSECRETARITH_TD_

include "mlir/Pass/PassBase.td"

def TosaToSecretArith : Pass<"tosa-to-secret-arith"> {
  let summary = "Lower `tosa.sigmoid` to secret arith dialects.";

  let description = [{
    This pass lowers the `tosa.sigmoid` dialect to the polynomial approximation
    -0.004 * x^3 + 0.197 * x + 0.5 (composed of arith, affine, and tensor operations).

    This polynomial approximation of sigmoid only works over the range [-5, 5]
    and is taken from the paper ['Logisitic regression over encrypted data from
    fully homomorphic encryption' by Chen et al.](https://eprint.iacr.org/2018/462.pdf).
  }];
  let dependentDialects = [
    "mlir::heir::tensor_ext::TensorExtDialect",
  ];
}

#endif  // LIB_CONVERSION_TOSATOSECRETARITH_TOSATOSECRETARITH_TD_
