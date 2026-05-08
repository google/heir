// RUN: heir-opt --jaxite-ckks-parameter-selection %s | FileCheck %s

// CHECK: module attributes {jaxiteword.ckks_params = #jaxiteword.ckks_parameters<{{.*}}>}
!ct = !jaxiteword.ciphertext<2, 3, 4>
!ml = !jaxiteword.modulus_list<65536, 1152921504606844513, 1152921504606844417>

func.func @test_add(%ct1 : !ct, %ct2 : !ct, %modulus_list: !ml) -> !ct {
  %out = jaxiteword.add %ct1, %ct2, %modulus_list: (!ct, !ct, !ml) -> !ct
  return %out : !ct
}
