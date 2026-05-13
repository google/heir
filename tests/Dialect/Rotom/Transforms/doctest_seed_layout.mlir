// RUN: heir-opt %s --rotom-seed-layout=n=8 | FileCheck %s

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
