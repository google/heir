// RUN: heir-opt --mlir-print-local-scope --canonicalize --secret-to-ckks %s | FileCheck %s

!efi1 = !secret.secret<tensor<1024xf64>>

#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113], P = [36028797020209153], logDefaultScale = 45>} {
  // CHECK: func @test_eval_chebyshev
  func.func @test_eval_chebyshev(%arg0 : !efi1 {mgmt.mgmt = #mgmt}) -> (!efi1 {mgmt.mgmt = #mgmt}) {
    // CHECK-NOT: secret.generic
    // CHECK: kernel.eval_chebyshev
    // CHECK-SAME: coefficients = [1.000000e+00, 2.000000e+00]
    // CHECK-SAME: lwe_ciphertext
    %0 = secret.generic(%arg0: !efi1) {
      ^bb0(%ARG0 : tensor<1024xf64>):
        %1 = kernel.eval_chebyshev %ARG0 {
          coefficients = [1.0 : f64, 2.0 : f64]
        } : tensor<1024xf64> -> tensor<1024xf64>
        secret.yield %1 : tensor<1024xf64>
    } -> (!efi1 {mgmt.mgmt = #mgmt})
    return %0 : !efi1
  }
}
