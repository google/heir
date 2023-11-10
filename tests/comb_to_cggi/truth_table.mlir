// RUN: heir-opt --split-input-file --comb-to-cggi --cse %s | FileCheck %s

// CHECK-NOT: secret
// CHECK: @truth_table_all_secret([[ARG:%.*]]: [[LWET:!lwe.lwe_ciphertext<.*>]]) -> [[LWET]]
func.func @truth_table_all_secret(%arg0: !secret.secret<i1>) -> !secret.secret<i1> {
  // CHECK: [[VAL:%.+]] = cggi.lut3([[ARG]], [[ARG]], [[ARG]])
  %0 = secret.generic
      ins(%arg0:  !secret.secret<i1>) {
      ^bb0(%ARG0: i1) :
          %1 = comb.truth_table %ARG0, %ARG0, %ARG0 -> 6 : ui8
          secret.yield %1 : i1
      } -> (!secret.secret<i1>)
  // CHECK: return [[VAL]] : [[LWET]]
  func.return %0 : !secret.secret<i1>
}

// -----

// CHECK-NOT: secret
// CHECK: @truth_table_partial_secret([[ARG:%.*]]: [[LWET:!lwe.lwe_ciphertext<.*>]]) -> [[LWET]]
func.func @truth_table_partial_secret(%arg0: !secret.secret<i1>) -> !secret.secret<i1> {
  // CHECK: [[FALSE:%.+]] = arith.constant false
  %false = arith.constant false
  // CHECK: [[TRUE:%.+]] = arith.constant true
  %true = arith.constant true
  // CHECK: [[ENCFALSE:%.+]] = lwe.encode [[FALSE]]
  // CHECK: [[LWEFALSE:%.+]] = lwe.trivial_encrypt [[ENCFALSE]]
  // CHECK: [[ENCTRUE:%.+]] = lwe.encode [[TRUE]]
  // CHECK: [[LWETRUE:%.+]] = lwe.trivial_encrypt [[ENCTRUE]]
  // CHECK: [[VAL1:%.+]] = cggi.lut3([[LWEFALSE]], [[LWETRUE]], [[ARG]])
  // CHECK: [[VAL2:%.+]] = cggi.lut3([[LWEFALSE]], [[LWETRUE]], [[VAL1]])
  %0 = secret.generic
      ins(%false, %true, %arg0: i1, i1, !secret.secret<i1>) {
      ^bb0(%FALSE: i1, %TRUE: i1, %ARG0: i1) :
          %1 = comb.truth_table %FALSE, %TRUE, %ARG0 -> 6 : ui8
          %2 = comb.truth_table %FALSE, %TRUE, %1 -> 2 : ui8
          secret.yield %2 : i1
      } -> (!secret.secret<i1>)
  // CHECK: return [[VAL2]] : [[LWET]]
  func.return %0 : !secret.secret<i1>
}
