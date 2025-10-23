// RUN: heir-opt --implement-shift-network --canonicalize --cse --fold-plaintext-masks --cse %s | FileCheck %s

#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : exists (e0, e1, e2: i0 = 1 and ct = 0 and 16e2 = -i1 + slot + 16e0 and 0 <= i1 <= 31 and 0 <= slot <= 31 and 0 <= e1 <= 3 and -3 + i1 - 16e0 <= 4e1 <= i1 - 16e0) }">
module {
  // CHECK: @trivial_insert
  // CHECK-COUNT-3: arith.muli
  func.func @trivial_insert(%arg0: !secret.secret<tensor<2x32xi32>>) -> !secret.secret<tensor<2x32xi32>> {
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x32xi32>>) {
    ^body(%input0: tensor<2x32xi32>):
      %1 = tensor_ext.remap %input0 {permutation = #layout1} : tensor<2x32xi32>
      secret.yield %1 : tensor<2x32xi32>
    } -> !secret.secret<tensor<2x32xi32>>
    return %0 : !secret.secret<tensor<2x32xi32>>
  }
}
