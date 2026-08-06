// RUN: heir-opt --annotate-level %s | FileCheck %s

module {
  // Lattigo target test
  // CHECK: module @test_lattigo
  module @test_lattigo attributes {backend.lattigo} {
    func.func @test_lattigo_deg3(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
      %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
      ^body(%val: tensor<16xf32>):
        // coefficients of size 4 -> degree 3. Lattigo consumes 2 levels.
        // CHECK: kernel.eval_chebyshev
        // CHECK-SAME: mgmt.level = 2 : index
        %1 = kernel.eval_chebyshev %val {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64, 4.0 : f64]} : tensor<16xf32> -> tensor<16xf32>
        secret.yield %1 : tensor<16xf32>
      } -> !secret.secret<tensor<16xf32>>
      return %0 : !secret.secret<tensor<16xf32>>
    }

    func.func @test_lattigo_func(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
      %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
      ^body(%val: tensor<16xf32>):
        // coefficients of size 5 -> degree 4. std::bit_width(4) = 3 levels to drop.
        // CHECK: kernel.eval_chebyshev
        // CHECK-SAME: mgmt.level = 3 : index
        %1 = kernel.eval_chebyshev %val {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64, 4.0 : f64, 5.0 : f64]} : tensor<16xf32> -> tensor<16xf32>
        secret.yield %1 : tensor<16xf32>
      } -> !secret.secret<tensor<16xf32>>
      return %0 : !secret.secret<tensor<16xf32>>
    }
  }

  // OpenFHE target test
  // CHECK: module @test_openfhe
  module @test_openfhe attributes {backend.openfhe} {
    func.func @test_openfhe_deg3(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
      %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
      ^body(%val: tensor<16xf32>):
        // coefficients of size 4 -> degree 3. OpenFHE consumes 3 levels.
        // CHECK: kernel.eval_chebyshev
        // CHECK-SAME: mgmt.level = 3 : index
        %1 = kernel.eval_chebyshev %val {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64, 4.0 : f64]} : tensor<16xf32> -> tensor<16xf32>
        secret.yield %1 : tensor<16xf32>
      } -> !secret.secret<tensor<16xf32>>
      return %0 : !secret.secret<tensor<16xf32>>
    }

    func.func @test_openfhe_func(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
      %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
      ^body(%val: tensor<16xf32>):
        // coefficients of size 3 -> degree 2. std::bit_width(2) = 2 levels to drop.
        // CHECK: kernel.eval_chebyshev
        // CHECK-SAME: mgmt.level = 2 : index
        %1 = kernel.eval_chebyshev %val {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64]} : tensor<16xf32> -> tensor<16xf32>
        secret.yield %1 : tensor<16xf32>
      } -> !secret.secret<tensor<16xf32>>
      return %0 : !secret.secret<tensor<16xf32>>
    }
  }

  // No backend target test (defaults to lattigo logic)
  // CHECK: module @test_default
  module @test_default {
    func.func @test_default_func(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
      %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
      ^body(%val: tensor<16xf32>):
        // CHECK: kernel.eval_chebyshev
        // CHECK-SAME: mgmt.level = 2 : index
        %1 = kernel.eval_chebyshev %val {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64, 4.0 : f64]} : tensor<16xf32> -> tensor<16xf32>
        secret.yield %1 : tensor<16xf32>
      } -> !secret.secret<tensor<16xf32>>
      return %0 : !secret.secret<tensor<16xf32>>
    }
  }
}
