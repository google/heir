// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.
module {
  // CHECK-LABEL: func @test
  func.func @test(
     %arg_cc: !openfhe.crypto_context,
     %arg_pk: !openfhe.public_key) {
    return
  }
}
