// RUN: heir-opt --openfhe-configure-crypto-context=entry-function=simple_sum %s | FileCheck %s

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177, 67239937, 66961409, 66813953], P = [536952833, 536690689], logDefaultScale = 26>, scheme.ckks} {
  func.func @simple_sum(%arg0: !openfhe.crypto_context, %arg1: !openfhe.ciphertext) -> !openfhe.ciphertext {
    %0 = openfhe.mod_reduce %arg0, %arg1 : (!openfhe.crypto_context, !openfhe.ciphertext) -> !openfhe.ciphertext
    return %0 : !openfhe.ciphertext
  }

  // CHECK: @simple_sum__generate_crypto_context
  // CHECK: openfhe.gen_params
  // CHECK-SAME: batchSize = 4096
  // CHECK-SAME: firstModSize = 30
  // CHECK-SAME: insecure = true
  // CHECK-SAME: mulDepth = 5
  // CHECK-SAME: plainMod = 0
  // CHECK-SAME: ringDim = 8192
  // CHECK-SAME: scalingModSize = 26
  // CHECK-SAME: scalingTechniqueFixedManual = true
  // CHECK: @simple_sum__configure_crypto_context
}
