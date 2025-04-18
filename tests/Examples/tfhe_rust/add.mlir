// This test ensures the testing harness is working properly with minimal codegen.

!sks = !tfhe_rust.server_key
!lut = !tfhe_rust.lookup_table
!eui3 = !tfhe_rust.eui3

func.func @fn_under_test(%sks : !sks, %a: !eui3, %b: !eui3) -> !eui3 {
  %res = tfhe_rust.add %sks, %a, %b: (!sks, !eui3, !eui3) -> !eui3
  return %res : !eui3
}
