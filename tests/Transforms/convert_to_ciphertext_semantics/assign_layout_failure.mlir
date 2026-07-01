// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=32 --verify-diagnostics

#layout3 = dense<[[0, 0, 0, 7], [0, 18, 0, 2], [0, 2, 0, 5], [0, 3, 0, 4], [0, 4, 0, 3], [0, 5, 0, 2], [0, 6, 0, 1], [0, 7, 0, 0]]> : tensor<8x4xi64>
module {
  func.func @permutate_vector_error() {
    %cst = arith.constant dense<1> : tensor<16xi16>
    %0 = secret.generic() {
      // expected-error@+2 {{Permutation index out of bounds: src_ct=0, src_slot=18 (input bounds: ct < 1, slot < 16); dst_ct=0, dst_slot=2 (target bounds: ct < 1, slot < 32)}}
      // expected-error@+1 {{'tensor_ext.assign_layout' op failed to verify that all of {value, output} have same type}}
      %1 = tensor_ext.assign_layout %cst {layout = #layout3, tensor_ext.layout = #layout3} : tensor<16xi16>
      secret.yield %1 : tensor<16xi16>
    } -> (!secret.secret<tensor<16xi16>> {tensor_ext.layout = #layout3})
    return
  }
}
