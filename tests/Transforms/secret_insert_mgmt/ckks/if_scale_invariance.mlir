// RUN: heir-opt "--secret-insert-mgmt-ckks=after-mul=false before-mul-include-first-mul=false level-budget=2 slot-number=1024" %s | FileCheck %s

// CHECK: func.func @test_scf_if_scale_mismatch
// CHECK: secret.generic
// CHECK:   %[[RES:.*]] = scf.if
// CHECK:     %[[MUL:.*]] = arith.mulf
// CHECK:     %[[RELINEARIZE:.*]] = mgmt.relinearize %[[MUL]]
// CHECK:     scf.yield %[[RELINEARIZE]]
// CHECK:   else
// CHECK:     %[[ADJUST_SCALE:.*]] = mgmt.adjust_scale %{{.*}}
// CHECK:     scf.yield %[[ADJUST_SCALE]]
module attributes {backend.lattigo, scheme.ckks} {
  func.func @test_scf_if_scale_mismatch(%cond: i1, %arg0: !secret.secret<tensor<1xf32>>, %arg1: !secret.secret<tensor<1xf32>>) -> !secret.secret<tensor<1xf32>> {
    %res = secret.generic(%arg0: !secret.secret<tensor<1xf32>>, %arg1: !secret.secret<tensor<1xf32>>) {
    ^body(%input_acc: tensor<1xf32>, %slice: tensor<1xf32>):
      %if_res = scf.if %cond -> (tensor<1xf32>) {
        %mul = arith.mulf %input_acc, %slice : tensor<1xf32>
        scf.yield %mul : tensor<1xf32>
      } else {
        scf.yield %input_acc : tensor<1xf32>
      }
      secret.yield %if_res : tensor<1xf32>
    } -> !secret.secret<tensor<1xf32>>
    return %res : !secret.secret<tensor<1xf32>>
  }
}
