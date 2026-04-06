// RUN: heir-opt "--secret-insert-mgmt-ckks=after-mul=true before-mul-include-first-mul=false bootstrap-waterline=10 level-budget=2 slot-number=1024" %s | FileCheck %s

// CHECK: func.func @matvec
// CHECK-SAME: {mgmt.mgmt = #mgmt.mgmt<level = 1>}
// CHECK-SAME: -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})

// CHECK: secret.generic
// CHECK: ^body([[INPUT:%.*]]: tensor<1x1024xf32>):

// CHECK: scf.if {{.*}} -> (tensor<1x1024xf32>) {
// CHECK:   mgmt.modreduce
// CHECK:   scf.yield
// CHECK: } else {
// CHECK-NEXT: mgmt.level_reduce [[INPUT]]
// CHECK-SAME: #mgmt.mgmt<level = 0>
// CHECK:   scf.yield
// CHECK: } {mgmt.mgmt = #mgmt.mgmt<level = 0>}

// CHECK: secret.yield {{.*}} : tensor<1x1024xf32>
module attributes {backend.lattigo, scheme.ckks} {
  func.func @matvec(%arg0: !secret.secret<tensor<1x1024xf32>>, %arg1: tensor<512x1024xf32>, %arg2: index, %arg3: index) -> !secret.secret<tensor<1x1024xf32>> {
    %c23 = arith.constant 23 : index
    %c-23 = arith.constant -23 : index
    %c512 = arith.constant 512 : index
    %0 = arith.muli %arg2, %c23 : index
    %1 = arith.addi %arg3, %0 : index
    %2 = arith.cmpi slt, %1, %c512 : index
    %3 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %4 = scf.if %2 -> (tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %arg1[%1, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1x1024xf32>
        %5 = arith.muli %arg2, %c-23 : index
        %6 = tensor_ext.rotate %extracted_slice, %5 : tensor<1x1024xf32>, index
        %7 = tensor_ext.rotate %input0, %arg3 : tensor<1x1024xf32>, index
        %8 = arith.mulf %6, %7 : tensor<1x1024xf32>
        %9 = arith.addf %input0, %8 : tensor<1x1024xf32>
        scf.yield %9 : tensor<1x1024xf32>
      } else {
        scf.yield %input0 : tensor<1x1024xf32>
      }
      secret.yield %4 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %3 : !secret.secret<tensor<1x1024xf32>>
  }
}
