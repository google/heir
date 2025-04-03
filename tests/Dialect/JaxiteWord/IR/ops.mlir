// RUN: heir-opt %s | FileCheck %s

!ct = !jaxiteword.ciphertext<2, 3, 4>
!ml = !jaxiteword.modulus_list<65536, 1152921504606844513, 1152921504606844417>

// CHECK: func.func @test_add(
func.func @test_add(%ct1 : !ct, %ct2 : !ct, %modulus_list: !ml) -> !ct {
  %out = jaxiteword.add %ct1, %ct2, %modulus_list: (!ct, !ct, !ml) -> !ct
  return %out : !ct
}
