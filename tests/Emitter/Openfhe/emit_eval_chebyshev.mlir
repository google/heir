// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

// CHECK: CiphertextT test_eval_chebyshev(
// CHECK-SAME:    CryptoContextT [[CC:[^,]*]],
// CHECK-SAME:    CiphertextT [[ARG1:[^)]*]]
// CHECK-SAME:  ) {
// CHECK-NEXT:      const auto& [[v1:.*]] = [[CC]]->EvalChebyshevSeries([[ARG1]], std::vector<double>{1, 2.5, 3}, -1, 1);
// CHECK-NEXT:      return [[v1]];
// CHECK-NEXT:  }
module attributes {scheme.ckks} {
  func.func @test_eval_chebyshev(%cc : !cc, %input : !ct) -> !ct {
    %res = openfhe.eval_chebyshev_series %cc, %input {
      coefficients = [1.0 : f64, 2.5 : f64, 3.0 : f64],
      domain_lower = -1.0 : f64,
      domain_upper = 1.0 : f64
    } : (!cc, !ct) -> !ct
    return %res : !ct
  }
}
