// This test ensures the testing harness is working properly with minimal codegen.

!sks = !tfhe_rust.server_key
!eui3 = !tfhe_rust.eui3

func.func @fn_under_test(%sks : !sks, %a: !eui3, %b: !eui3) -> !eui3 {
  %res = tfhe_rust.bitand %sks, %a, %b: (!sks, !eui3, !eui3) -> !eui3
  return %res : !eui3
}
