// RUN: heir-opt --secret-insert-mgmt-ckks="after-mul=true before-mul-include-first-mul=false bootstrap-waterline=40 level-budget=40 slot-number=8" %s | FileCheck %s

module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: func.func @main
  func.func @main(%arg0: !secret.secret<tensor<8xf32>>, %arg1: !secret.secret<tensor<2x8xf32>>) -> !secret.secret<tensor<2x8xf32>> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = secret.generic(%arg0 : !secret.secret<tensor<8xf32>>, %arg1 : !secret.secret<tensor<2x8xf32>>) {
    ^body(%input0: tensor<8xf32>, %input1: tensor<2x8xf32>):
      %1 = scf.for %i = %c1 to %c2 step %c1 iter_args(%iter = %input1) -> (tensor<2x8xf32>) {
        %m = arith.mulf %input0, %input0 : tensor<8xf32>
        %m_2d = tensor.expand_shape %m [[0, 1]] output_shape [1, 8] : tensor<8xf32> into tensor<1x8xf32>
        // CHECK: %[[bootstrap:.*]] = mgmt.bootstrap
        // CHECK: %[[m:.*]] = arith.mulf
        // CHECK: %[[relin:.*]] = mgmt.relinearize %[[m]]
        // CHECK: %[[reduced_slice:.*]] = mgmt.modreduce %[[relin]]
        // CHECK: %[[expanded:.*]] = tensor.expand_shape %[[reduced_slice]]
        // CHECK: %[[adjusted_dest:.*]] = mgmt.adjust_scale %[[bootstrap]]
        // CHECK: %[[reduced_dest:.*]] = mgmt.modreduce %[[adjusted_dest]]
        // CHECK: tensor.insert_slice %[[expanded]] into %[[reduced_dest]]
        %updated = tensor.insert_slice %m_2d into %iter[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> into tensor<2x8xf32>
        scf.yield %updated : tensor<2x8xf32>
      }
      secret.yield %1 : tensor<2x8xf32>
    } -> !secret.secret<tensor<2x8xf32>>
    return %0 : !secret.secret<tensor<2x8xf32>>
  }
}
