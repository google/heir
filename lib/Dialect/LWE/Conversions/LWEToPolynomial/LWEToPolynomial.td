#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_

include "mlir/Pass/PassBase.td"

def LWEToPolynomial : Pass<"lwe-to-polynomial"> {
  let summary = "Lower `lwe` to `polynomial` dialect.";

  let description = [{
    This pass lowers the `lwe` dialect to `polynomial` dialect, making the
    polynomial arithmetic behind each ciphertext operation explicit.

    Types are lowered as follows:

    - An RLWE ciphertext made of `N` polynomials, e.g., `(c_0, c_1)`, becomes
      a `tensor<N x !polynomial.polynomial<...>>` over the ciphertext ring.
      Tensors of ciphertexts become higher-dimensional tensors of polynomials.
    - Plaintexts and ring elements become a single polynomial over their ring.
    - Secret keys become a tensor of one polynomial `s`, and public keys a
      tensor of two polynomials `(b, a) = (-a * s + e, a)`.

    A ciphertext `(c_0, c_1, ..., c_k)` represents the polynomial
    `c_0 + c_1 * s + ... + c_k * s^k` in the secret key `s`, so decrypting a
    two-polynomial ciphertext computes `c_0 + c_1 * s`. Encryption samples its
    randomness with the `random` dialect, and how the plaintext and noise are
    combined depends on the ciphertext's encryption type (see
    `LweEncryptionType`). The plaintext and ciphertext rings must be related
    as that encryption type expects: e.g., CKKS plaintexts must already be
    encoded into the ciphertext's ring.

    (* example filepath=tests/Dialect/LWE/Conversions/lwe_to_polynomial/doctest.mlir *)
  }];

  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::random::RandomDialect"
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_
