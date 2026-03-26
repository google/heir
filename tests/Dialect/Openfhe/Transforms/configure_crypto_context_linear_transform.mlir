// RUN: heir-opt --openfhe-configure-crypto-context=entry-function=linear_transform %s | FileCheck %s

module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177, 67239937, 66961409, 66813953], P = [536952833, 536690689], logDefaultScale = 26>} {
  func.func @linear_transform(%arg0: !openfhe.crypto_context, %arg1: !openfhe.ciphertext, %arg2: tensor<2x4096xf64>) -> !openfhe.ciphertext {
    %0 = openfhe.linear_transform %arg0, %arg1, %arg2 {diagonal_indices = array<i32: 0, 1>, logBabyStepGiantStepRatio = 2 : i64} : (!openfhe.crypto_context, !openfhe.ciphertext, tensor<2x4096xf64>) -> !openfhe.ciphertext
    return %0 : !openfhe.ciphertext
  }

  // CHECK: @linear_transform__generate_crypto_context
  // CHECK: openfhe.gen_params
  // CHECK-SAME: mulDepth = 5
  // CHECK: openfhe.gen_context %{{.*}} {supportFHE = false}
  // CHECK: @linear_transform__configure_crypto_context
  // CHECK-NOT: openfhe.gen_mulkey
  // CHECK: openfhe.gen_rotkey
}
