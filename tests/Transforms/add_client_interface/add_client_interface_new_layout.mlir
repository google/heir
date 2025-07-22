// TODO(#2047): re-enable
// CHANGE BACK: heir-opt --add-client-interface="ciphertext-size=1024" %s | FileCheck %s
// RUN: heir-opt %s

// Data is 32x64, being packed into ciphertexts of size 1024 via Halevi-Shoup
// diagonal layout.
!ct_ty = !secret.secret<tensor<32x1024xi16>>
#layout = #tensor_ext.new_layout<domainSize=2, relation="(row, col, ct, slot) : ((slot mod 32) - row == 0, (ct + slot) mod 64 - col == 0, row >= 0, col >= 0, ct >= 0, slot >= 0, 1023 - slot >= 0, 31 - ct >= 0, 31 - row >= 0, 64 - col >= 0)">
#original_type = #tensor_ext.original_type<originalType = tensor<32x64xi16>, layout = #layout>

func.func @add(
    %arg0: !ct_ty {tensor_ext.original_type = #original_type}
) -> (!ct_ty {tensor_ext.original_type = #original_type}) {
  %0 = secret.generic(%arg0: !ct_ty) {
  ^body(%pt_arg0: tensor<32x1024xi16>):
    %0 = arith.addi %pt_arg0, %pt_arg0 : tensor<32x1024xi16>
    secret.yield %0 : tensor<32x1024xi16>
  } -> !ct_ty
  return %0 : !ct_ty
}
