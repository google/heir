// RUN: heir-opt --secret-distribute-generic %s | FileCheck %s

// CHECK: func.func @test
// CHECK: %[[LOOP:.*]]:2 = scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}, %[[SHIFT:.*]] = %{{.*}}) -> (!secret.secret<tensor<1x1024xf32>>, index)
// CHECK: %[[GEN1:.*]] = secret.generic(%[[ACC]]: !secret.secret<tensor<1x1024xf32>>)
// CHECK: ^body(%[[INPUT1:.*]]: tensor<1x1024xf32>):
// CHECK: %[[ROT:.*]] = tensor_ext.rotate %[[INPUT1]], %[[SHIFT]]
// CHECK: secret.yield %[[ROT]]
// CHECK: %[[GEN2:.*]] = secret.generic(%[[ACC]]: !secret.secret<tensor<1x1024xf32>>, %[[GEN1]]: !secret.secret<tensor<1x1024xf32>>)
// CHECK: ^body(%[[INPUT2:.*]]: tensor<1x1024xf32>, %[[INPUT3:.*]]: tensor<1x1024xf32>):
// CHECK: %[[ADD:.*]] = arith.addf %[[INPUT2]], %[[INPUT3]]
// CHECK: secret.yield %[[ADD]]
// CHECK: %[[NEXT_SHIFT:.*]] = arith.divsi %[[SHIFT]], %{{.*}}
// CHECK: scf.yield %[[GEN2]], %[[NEXT_SHIFT]] : !secret.secret<tensor<1x1024xf32>>, index
// CHECK: return %[[LOOP]]#0

module {
  func.func @test(%arg0: !secret.secret<tensor<1x1024xf32>>, %arg1: index) -> !secret.secret<tensor<1x1024xf32>> {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %res = secret.generic(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input: tensor<1x1024xf32>):
      %loop:2 = scf.for %iv = %c0 to %c5 step %c1 iter_args(%acc = %input, %shift = %arg1) -> (tensor<1x1024xf32>, index) {
        %rot = tensor_ext.rotate %acc, %shift : tensor<1x1024xf32>, index
        %add = arith.addf %acc, %rot : tensor<1x1024xf32>
        %next_shift = arith.divsi %shift, %c2 : index
        scf.yield %add, %next_shift : tensor<1x1024xf32>, index
      }
      secret.yield %loop#0 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>>)
    return %res : !secret.secret<tensor<1x1024xf32>>
  }
}
