// RUN: heir-opt --tosa-to-boolean-jaxite="parallelism=4" %s | heir-translate --emit-jaxite | FileCheck %s

module {
  // CHECK-LABEL: def test_add_one_lut3(
  // CHECK-NEXT:   [[v0:.*]]: list[types.LweCiphertext],
  // CHECK-NEXT:   [[v1:.*]]: jaxite_bool.ServerKeySet,
  // CHECK-NEXT:   [[v2:.*]]: jaxite_bool.Parameters,
  // CHECK-NEXT: ) -> list[types.LweCiphertext]:
  // CHECK-COUNT-1: jaxite_bool.constant
  // CHECK-COUNT-2: jaxite_bool.pmap_lut3
  func.func @test_add_one_lut3(%in: i8 {secret.secret}) -> (i8) {
    %1 = arith.constant 1 : i8
    %2 = arith.addi %in, %1 : i8
    return %2 : i8
  }
}
