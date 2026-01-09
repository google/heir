// RUN: heir-opt --openfhe-configure-crypto-context=entry-function=bootstrap %s | FileCheck %s

!ct = !openfhe.ciphertext

func.func @bootstrap(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  %0 = openfhe.bootstrap %arg0, %arg1 : (!openfhe.crypto_context, !ct) -> !ct
  return %0 : !ct
}

// CHECK: @bootstrap
// CHECK: @bootstrap__generate_crypto_context
// CHECK: mulDepth = 20
// CHECK: openfhe.gen_context %{{.*}} {supportFHE = true}

// CHECK: @bootstrap__configure_crypto_context
// CHECK: openfhe.gen_mulkey
// CHECK: openfhe.setup_bootstrap %{{.*}} {levelBudgetDecode = 3 : index, levelBudgetEncode = 3 : index}
// CHECK: openfhe.gen_bootstrapkey
