// RUN: heir-opt --secret-insert-mgmt-ckks="bootstrap-waterline=0 level-budget=15 after-mul=true min-slot-count=4096" %s | FileCheck %s

// CHECK-LABEL: @repro
// CHECK: %[[BOOT:.*]] = mgmt.bootstrap
// CHECK-NEXT: %[[ADJ:.*]] = mgmt.adjust_scale %[[BOOT]]
// CHECK-NEXT: mgmt.modreduce %[[ADJ]]
module {
  func.func @repro(%arg0: !secret.secret<tensor<1xf32>>) -> !secret.secret<tensor<1xf32>> {
    %res = secret.generic(%arg0 : !secret.secret<tensor<1xf32>>) {
    ^bb0(%x: tensor<1xf32>):
      %r = mgmt.modreduce %x : tensor<1xf32>
      secret.yield %r : tensor<1xf32>
    } -> !secret.secret<tensor<1xf32>>
    return %res : !secret.secret<tensor<1xf32>>
  }
}
