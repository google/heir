// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.
module {
  // CHECK-LABEL: func @test
  func.func @test(
     %arg_eui2: !tfhe_rs.eui2,
     %arg_eui3: !tfhe_rs.eui3,
     %arg_eui4: !tfhe_rs.eui4,
     %arg_eui8: !tfhe_rs.eui8,
     %arg_eui10: !tfhe_rs.eui10,
     %arg_eui12: !tfhe_rs.eui12,
     %arg_eui14: !tfhe_rs.eui14,
     %arg_eui16: !tfhe_rs.eui16,
     %arg_eui32: !tfhe_rs.eui32,
     %arg_eui64: !tfhe_rs.eui64,
     %arg_eui128: !tfhe_rs.eui128,
     %arg_eui256: !tfhe_rs.eui256,
     %arg_ei8: !tfhe_rs.ei8,
     %arg_ei16: !tfhe_rs.ei16,
     %arg_ei32: !tfhe_rs.ei32,
     %arg_ei64: !tfhe_rs.ei64,
     %arg_ei128: !tfhe_rs.ei128,
     %arg_ei256: !tfhe_rs.ei256) {
    return
  }
}
