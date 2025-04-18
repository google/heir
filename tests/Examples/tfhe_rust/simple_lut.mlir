// This test ensures the testing harness is working properly with minimal codegen.

!sks = !tfhe_rust.server_key
!lut = !tfhe_rust.lookup_table
!eui3 = !tfhe_rust.eui3

func.func @fn_under_test(%sks : !sks, %a: !eui3, %b: !eui3) -> !eui3 {
  %lut = tfhe_rust.generate_lookup_table %sks {truthTable = 7 : ui8} : (!sks) -> !lut
  %0 = tfhe_rust.scalar_left_shift %sks, %a {shiftAmount = 1 : index} : (!sks, !eui3) -> !eui3
  %1 = tfhe_rust.add %sks, %0, %b : (!sks, !eui3, !eui3) -> !eui3
  %2 = tfhe_rust.apply_lookup_table %sks, %1, %lut : (!sks, !eui3, !lut) -> !eui3
  return %2 : !eui3
}
