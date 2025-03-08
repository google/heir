// RUN: heir-translate --emit-jaxite %s | FileCheck %s

!ct = !jaxiteword.ciphertext<2, 3, 4>
!ml = !jaxiteword.modulus_list<i32, i32, i32>

// CHECK-LABEL: func.func @test_add(
func.func @test_add(%ct1 : !ct, %ct2 : !ct, %modulus_list: !ml) -> !ct {
  // find functions here: third_party/heir/tests/Dialect/Openfhe/IR/ops.mlir
  // ToDo: How to create value for all inputs?
  %out = jaxiteword.add %ct1, %ct2, %modulus_list: (!ct, !ct, !ml) -> !ct
  return %out : !ct
}
