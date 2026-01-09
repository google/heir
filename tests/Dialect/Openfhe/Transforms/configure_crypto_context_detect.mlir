// RUN: heir-opt --openfhe-configure-crypto-context --split-input-file %s | FileCheck %s

// Test whether detection works for one main function

!ct = !openfhe.ciphertext

func.func @simple_sum(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  %14 = openfhe.mod_reduce %arg0, %arg1 : (!openfhe.crypto_context, !ct) -> !ct
  return %14 : !ct
}

// CHECK: @simple_sum
// CHECK: @simple_sum__generate_crypto_context
// CHECK: @simple_sum__configure_crypto_context

// -----

// Test whether called function is skipped

!ct = !openfhe.ciphertext

func.func @test(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  return %arg1 : !ct
}

func.func @simple_sum(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  %0 = openfhe.mod_reduce %arg0, %arg1 : (!openfhe.crypto_context, !ct) -> !ct
  %1 = call @test(%arg0, %0) : (!openfhe.crypto_context, !ct) -> !ct
  return %1 : !ct
}

// CHECK: @test
// CHECK: @simple_sum
// CHECK: @simple_sum__generate_crypto_context
// CHECK: @simple_sum__configure_crypto_context
