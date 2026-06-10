// RUN: heir-opt --annotate-level %s | FileCheck %s
module {
  // CHECK: func.func @test_openfhe
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !{{[a-zA-Z0-9_.]+}}
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: !{{[a-zA-Z0-9_.]+}} {mgmt.level = 0 : index}
  // CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]+]]: !{{[a-zA-Z0-9_.]+}} {mgmt.level = 0 : index}
  func.func @test_openfhe(%cc: !openfhe.crypto_context, %ct: !openfhe.ciphertext, %pt: !openfhe.plaintext) -> !openfhe.ciphertext {
    // CHECK: openfhe.mod_reduce
    // CHECK-SAME: mgmt.level = 1 : index
    %ct_reduced = openfhe.mod_reduce %cc, %ct : (!openfhe.crypto_context, !openfhe.ciphertext) -> !openfhe.ciphertext

    // CHECK: openfhe.level_reduce
    // CHECK-SAME: mgmt.level = 3 : index
    %ct_reduced2 = openfhe.level_reduce %cc, %ct_reduced {levelToDrop = 2 : i64} : (!openfhe.crypto_context, !openfhe.ciphertext) -> !openfhe.ciphertext

    // CHECK: openfhe.bootstrap
    // CHECK-SAME: mgmt.level = 0 : index
    %ct_bootstrapped = openfhe.bootstrap %cc, %ct_reduced2 : (!openfhe.crypto_context, !openfhe.ciphertext) -> !openfhe.ciphertext

    // CHECK: openfhe.mul_plain
    // CHECK-SAME: mgmt.level = 0 : index
    %ct_mul = openfhe.mul_plain %cc, %ct_bootstrapped, %pt : (!openfhe.crypto_context, !openfhe.ciphertext, !openfhe.plaintext) -> !openfhe.ciphertext

    return %ct_mul : !openfhe.ciphertext
  }
}
