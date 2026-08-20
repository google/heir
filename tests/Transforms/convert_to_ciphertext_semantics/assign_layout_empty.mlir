// RUN: heir-opt %s --convert-to-ciphertext-semantics=min-slot-count=32 | FileCheck %s

// An accumulator whose init is tensor.empty must be zero-filled rather than
// left as an undefined ciphertext tensor: an undefined ciphertext has no
// runtime representation, and a rolled kernel writes the elements one at a
// time, so a pass that consumes the whole tensor - bootstrap placement does -
// would read the elements the loop has not reached yet.

#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 15 and 0 <= slot <= 31 }">

// CHECK: func.func @empty_accumulator_is_zero_filled
// CHECK-NOT: tensor.empty
// CHECK: arith.constant dense<0>
// CHECK-SAME: tensor<1x32xi16>
// CHECK-NOT: tensor.empty
module {
  func.func @empty_accumulator_is_zero_filled() {
    %empty = tensor.empty() : tensor<16xi16>
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %empty {layout = #layout, tensor_ext.layout = #layout} : tensor<16xi16>
      secret.yield %1 : tensor<16xi16>
    } -> (!secret.secret<tensor<16xi16>> {tensor_ext.layout = #layout})
    return
  }
}
