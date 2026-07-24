// RUN: heir-opt --secret-insert-mgmt-ckks="bootstrap-waterline=2 level-budget=2 slot-number=8 after-mul=true" %s | FileCheck %s

module {
  // CHECK: func.func @main
  func.func @main(%arg0: tensor<1x8xf32> {secret.secret}) -> tensor<1x8xf32> {
    %cst = arith.constant dense<2.000000e+00> : tensor<1x8xf32>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // First loop: yields a value forced to FHE level 0.
    %loop1 = scf.for %i = %c1 to %c3 step %c1 iter_args(%iter1 = %arg0) -> (tensor<1x8xf32>) {
      %m1 = arith.mulf %iter1, %cst : tensor<1x8xf32>
      scf.yield %m1 : tensor<1x8xf32>
    }

    // Second loop: takes the level 0 value, multiplies it.
    // CHECK: mgmt.bootstrap
    // CHECK: arith.mulf
    // CHECK: mgmt.modreduce
    // CHECK: arith.mulf
    // CHECK: mgmt.modreduce
    // CHECK: mgmt.bootstrap
    // CHECK: arith.mulf
    // CHECK: mgmt.modreduce
    %loop2 = scf.for %j = %c1 to %c2 step %c1 iter_args(%iter2 = %loop1) -> (tensor<1x8xf32>) {
      %m2 = arith.mulf %iter2, %cst : tensor<1x8xf32>
      scf.yield %m2 : tensor<1x8xf32>
    }

    return %loop2 : tensor<1x8xf32>
  }
}
