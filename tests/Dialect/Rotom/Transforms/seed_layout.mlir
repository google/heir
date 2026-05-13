// RUN: heir-opt %s --rotom-seed-layout=n=8 --split-input-file | FileCheck %s

module {
  // CHECK: func.func @test_seeding(%{{.*}}: !secret.secret<tensor<4x4xf32>> {rotom.seed = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 4>, #rotom.dim<dim = 0, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 2>, #rotom.dim<dim = 0, size = 4>], n = 8>]>}, %{{.*}}: tensor<4x4xf32> {rotom.seed = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 4>, #rotom.dim<dim = 0, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 2>, #rotom.dim<dim = 0, size = 4>], n = 8>]>})
  func.func @test_seeding(%arg0: !secret.secret<tensor<4x4xf32>>, %arg1: tensor<4x4xf32>) -> !secret.secret<tensor<4x4xf32>> {
    // CHECK: secret.generic(%{{.*}}: !secret.secret<tensor<4x4xf32>>)
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
  // CHECK: func.func @test_seeding_3d(%{{.*}}: !secret.secret<tensor<2x2x2xf32>> {rotom.seed = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 2>, #rotom.dim<dim = 2, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 2, size = 2>, #rotom.dim<dim = 1, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2>, #rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 2, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2>, #rotom.dim<dim = 2, size = 2>, #rotom.dim<dim = 0, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 2, size = 2>, #rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 2, size = 2>, #rotom.dim<dim = 1, size = 2>, #rotom.dim<dim = 0, size = 2>], n = 8>]>})
  func.func @test_seeding_3d(%arg0: !secret.secret<tensor<2x2x2xf32>>) -> !secret.secret<tensor<2x2x2xf32>> {
    // CHECK: secret.generic(%{{.*}}: !secret.secret<tensor<2x2x2xf32>>)
    %0 = secret.generic(%arg0 : !secret.secret<tensor<2x2x2xf32>>) {
    ^bb0(%arg1: tensor<2x2x2xf32>):
      secret.yield %arg1 : tensor<2x2x2xf32>
    } -> !secret.secret<tensor<2x2x2xf32>>
    return %0 : !secret.secret<tensor<2x2x2xf32>>
  }
}

// -----

module {
  // CHECK: func.func @test_seeding_non_pow2(%{{.*}}: !secret.secret<tensor<3x3xf32>> {rotom.seed = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 4>, #rotom.dim<dim = 0, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 2>, #rotom.dim<dim = 0, size = 4>], n = 8>]>}, %{{.*}}: tensor<3x3xf32> {rotom.seed = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 4>, #rotom.dim<dim = 0, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 2>], n = 8>, #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 2>, #rotom.dim<dim = 0, size = 4>], n = 8>]>})
  func.func @test_seeding_non_pow2(%arg0: !secret.secret<tensor<3x3xf32>>, %arg1: tensor<3x3xf32>) -> !secret.secret<tensor<3x3xf32>> {
    // CHECK: secret.generic(%{{.*}}: !secret.secret<tensor<3x3xf32>>)
    %0 = secret.generic(%arg0 : !secret.secret<tensor<3x3xf32>>) {
    ^bb0(%arg2: tensor<3x3xf32>):
      %1 = arith.addf %arg2, %arg1 : tensor<3x3xf32>
      secret.yield %1 : tensor<3x3xf32>
    } -> !secret.secret<tensor<3x3xf32>>
    return %0 : !secret.secret<tensor<3x3xf32>>
  }
}
