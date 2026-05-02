// RUN: heir-opt %s --rotom-seed-layout=n=8 --split-input-file | FileCheck %s

#d0 = #rotom.dim<dim = 0, size = 2, stride = 2>
#d1 = #rotom.dim<dim = 0, size = 2, stride = 1>
#d2 = #rotom.dim<dim = 1, size = 4, stride = 1>
#layout1 = #rotom.layout<dims = [#d0, #d1, #d2], n = 8>

#d3 = #rotom.dim<dim = 1, size = 4, stride = 1>
#d4 = #rotom.dim<dim = 0, size = 2, stride = 1>
#layout2 = #rotom.layout<dims = [#d0, #d3, #d4], n = 8>

#d5 = #rotom.dim<dim = 1, size = 2, stride = 2>
#d6 = #rotom.dim<dim = 1, size = 2, stride = 1>
#d7 = #rotom.dim<dim = 0, size = 4, stride = 1>
#layout3 = #rotom.layout<dims = [#d5, #d6, #d7], n = 8>

#d8 = #rotom.dim<dim = 0, size = 4, stride = 1>
#d9 = #rotom.dim<dim = 1, size = 2, stride = 1>
#layout4 = #rotom.layout<dims = [#d5, #d8, #d9], n = 8>

module {
  // CHECK: func.func @test_seeding(%{{.*}}, %{{.*}} {rotom.seed = #seed})
  func.func @test_seeding(%arg0: !secret.secret<tensor<4x4xf32>>, %arg1: tensor<4x4xf32>) -> !secret.secret<tensor<4x4xf32>> {
    // CHECK: secret.generic(%{{.*}}: !secret.secret<tensor<4x4xf32>> {rotom.seed = #seed})
    %0 = secret.generic(%arg0 : !secret.secret<tensor<4x4xf32>>) {
    ^bb0(%arg2: tensor<4x4xf32>):
      %1 = arith.addf %arg2, %arg1 : tensor<4x4xf32>
      secret.yield %1 : tensor<4x4xf32>
    } -> !secret.secret<tensor<4x4xf32>>
    return %0 : !secret.secret<tensor<4x4xf32>>
  }
}

// -----

module {
  // CHECK: func.func @test_seeding_3d
  func.func @test_seeding_3d(%arg0: !secret.secret<tensor<2x2x2xf32>>) -> !secret.secret<tensor<2x2x2xf32>> {
    // CHECK: secret.generic(%{{.*}}: !secret.secret<tensor<2x2x2xf32>> {rotom.seed = #seed})
    %0 = secret.generic(%arg0 : !secret.secret<tensor<2x2x2xf32>>) {
    ^bb0(%arg1: tensor<2x2x2xf32>):
      secret.yield %arg1 : tensor<2x2x2xf32>
    } -> !secret.secret<tensor<2x2x2xf32>>
    return %0 : !secret.secret<tensor<2x2x2xf32>>
  }
}
