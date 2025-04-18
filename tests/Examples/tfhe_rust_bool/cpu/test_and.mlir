// This test ensures the testing harness is working properly with minimal codegen.

!bsks = !tfhe_rust_bool.server_key
!eb = !tfhe_rust_bool.eb

func.func @fn_under_test(%bsks : !bsks, %a: !eb, %b: !eb) -> !eb {
  %res = tfhe_rust_bool.and %bsks, %a, %b: (!bsks, !eb, !eb) -> !eb
  return %res : !eb
}
