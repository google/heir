// RUN: heir-opt --openfhe-configure-crypto-context --split-input-file %s | FileCheck %s

// Test whether detection traverses nested functions

!ct = !openfhe.ciphertext

func.func @test(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  %0 = openfhe.bootstrap %arg0, %arg1 : (!openfhe.crypto_context, !ct) -> !ct
  return %0 : !ct
}

func.func @nested_bootstrap(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  %1 = call @test(%arg0, %arg1) : (!openfhe.crypto_context, !ct) -> !ct
  return %1 : !ct
}

// CHECK: @test
// CHECK: @nested_bootstrap
// CHECK: @nested_bootstrap__generate_crypto_context
// CHECK: @nested_bootstrap__configure_crypto_context
// CHECK: openfhe.gen_mulkey
// CHECK: openfhe.setup_bootstrap %{{.*}}
// CHECK: openfhe.gen_bootstrapkey

// -----

!ct = !openfhe.ciphertext

func.func @test(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  %0 = openfhe.mul %arg0, %arg1, %arg1 : (!openfhe.crypto_context, !ct, !ct) -> !ct
  return %0 : !ct
}

func.func @nested_mul(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  %1 = call @test(%arg0, %arg1) : (!openfhe.crypto_context, !ct) -> !ct
  return %1 : !ct
}

// CHECK: @test
// CHECK: @nested_mul
// CHECK: @nested_mul__generate_crypto_context
// CHECK: openfhe.gen_params
// CHECK-SAME: mulDepth = 1
// CHECK: @nested_mul__configure_crypto_context
// CHECK: openfhe.gen_mulkey
// CHECK-NOT: bootstrap


// -----

!ct = !openfhe.ciphertext

func.func @test(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  %0 = openfhe.rot %arg0, %arg1 {static_shift = 4 : index} : (!openfhe.crypto_context, !ct) -> !ct
  return %0 : !ct
}

func.func @nested_rot(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  %1 = call @test(%arg0, %arg1) : (!openfhe.crypto_context, !ct) -> !ct
  return %1 : !ct
}

// CHECK: @test
// CHECK: @nested_rot
// CHECK: @nested_rot__generate_crypto_context
// CHECK: openfhe.gen_params
// CHECK: @nested_rot__configure_crypto_context
// CHECK: openfhe.gen_rotkey
// CHECK-SAME: indices = array<i64: 4>
// CHECK-NOT: bootstrap
