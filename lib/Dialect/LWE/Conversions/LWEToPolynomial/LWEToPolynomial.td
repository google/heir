#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_

include "mlir/Pass/PassBase.td"

def LWEToPolynomial : Pass<"lwe-to-polynomial"> {
  let summary = "Lower `lwe` to `polynomial` dialect.";

  let description = [{
    This pass lowers the `lwe` dialect to `polynomial` dialect, implementing
    polynomial arithmetic behind each ciphertext operation.

    Types are lowered as follows:

    - An RLWE ciphertext made of `k` polynomials, e.g., `(c_0, c_1)`, becomes
      a `tensor<k x !polynomial.polynomial<...>>` over the ciphertext ring.
      Tensors of ciphertexts become higher-dimensional tensors of polynomials.
    - Plaintexts and ring elements become a single polynomial over their ring.
    - Secret keys become a tensor of one polynomial `s`, and public keys a
      tensor of two polynomials `(b, a) = (-a * s + e, a)`, where the noise
      `e` must be a multiple of the plaintext modulus for BGV.

    (* example filepath=tests/Dialect/LWE/Conversions/lwe_to_polynomial/doctest.mlir *)
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::mod_arith::ModArithDialect",
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::heir::random::RandomDialect",
    "mlir::heir::rns::RNSDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_
