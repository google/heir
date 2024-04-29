#ifndef LIB_DIALECT_BGV_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_BGV_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AddClientInterface : Pass<"bgv-add-client-interface"> {
  let summary = "Add client interfaces to BGV encrypted functions";
  let description = [{
  This pass adds encrypt and decrypt functions for each compiled function in the
  IR. These functions maintain the same interface as the original function,
  while the compiled function may lose some of this information by the lowerings
  to ciphertext types (e.g., a scalar ciphertext, when lowered through BGV, must
  be encoded as a tensor).

  Example:

  For an input function with signature

  ```mlir
  #encoding = ...
  #params = ...
  !in_ty = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type = tensor<32xi16>>
  !out_ty = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type = i16>
  func.func @my_func(%arg0: !in_ty) -> !out_ty {
    ...
  }
  ```

  The pass will generate two new functions with signatures

  ```mlir
  func.func @my_func__encrypt(
    %arg0: tensor<32xi16>,
    %sk: !lwe.rlwe_secret_key<...>
  ) -> !in_ty

  func.func @my_func__decrypt(
    %arg0: !out_ty,
    %sk: !lwe.rlwe_secret_key<...>
  ) -> i16
  ```

  The `my_func__encrypt` function has the same order of operands as `my_func`,
  and uses their `underylying_type` as the corresponding input type.
  The last operand is the encryption key.
  The same holds for `my_func__decrypt`, but the inputs are the return types
  of `my_func` and the results are the underlying types of the return types of `my_func`.

  If `use-public-key` is set to true, the encrypt function uses
  `lwe.rlwe_public_key` for encryption.

  If `one-value-per-helper-fn` is set to true, the encryption helpers are split
  into separate functions, one for each SSA value being converted. For example,
  using the same `!in_ty` and `!out_ty` as above, this function signature

  ```mlir
  func.func @my_func(%arg0: !in_ty, %arg1: !in_ty) -> (!out_ty, !out_ty)
  ```

  generates the following four helpers.

  ```mlir
  func.func @my_func__encrypt__arg0(%arg0: tensor<32xi16>, %sk: !lwe.rlwe_secret_key<...>) -> !in_ty
  func.func @my_func__encrypt__arg1(%arg1: tensor<32xi16>, %sk: !lwe.rlwe_secret_key<...>) -> !in_ty
  func.func @my_func__decrypt__result0(%arg0: !out_ty, %sk: !lwe.rlwe_secret_key<...>) -> i16
  func.func @my_func__decrypt__result1(%arg1: !out_ty, %sk: !lwe.rlwe_secret_key<...>) -> i16
  }
  ```

  The suffix `__argN` indicates the SSA value being encrypted is the N-th argument of `my_func`,
  and similarly for `__resultN`.
  }];
  let dependentDialects = ["mlir::heir::bgv::BGVDialect"];
  let options = [
    Option<"usePublicKey", "use-public-key", "bool", /*default=*/"false",
           "If true, generate a client interface that uses a public key for encryption.">,
    Option<"oneValuePerHelperFn", "one-value-per-helper-fn", "bool", /*default=*/"false",
           "If true, split encryption helpers into separate functions for each SSA value.">
  ];
}

#endif  // LIB_DIALECT_BGV_TRANSFORMS_PASSES_TD_
