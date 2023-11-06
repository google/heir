// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.
module {
  // CHECK-LABEL: func @test
  func.func @test(
     %arg_eui2: !tfhe_rust.eui2,
     %arg_eui3: !tfhe_rust.eui3,
     %arg_eui4: !tfhe_rust.eui4,
     %arg_eui8: !tfhe_rust.eui8,
     %arg_eui10: !tfhe_rust.eui10,
     %arg_eui12: !tfhe_rust.eui12,
     %arg_eui14: !tfhe_rust.eui14,
     %arg_eui16: !tfhe_rust.eui16,
     %arg_eui32: !tfhe_rust.eui32,
     %arg_eui64: !tfhe_rust.eui64,
     %arg_eui128: !tfhe_rust.eui128,
     %arg_eui256: !tfhe_rust.eui256,
     %arg_ei8: !tfhe_rust.ei8,
     %arg_ei16: !tfhe_rust.ei16,
     %arg_ei32: !tfhe_rust.ei32,
     %arg_ei64: !tfhe_rust.ei64,
     %arg_ei128: !tfhe_rust.ei128,
     %arg_ei256: !tfhe_rust.ei256) {
    return
  }
}
