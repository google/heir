// RUN: heir-opt --secret-distribute-generic --split-input-file --comb-to-cggi --cse %s | FileCheck %s

// CHECK-NOT: secret
// CHECK: @boolean_gates([[ARG:%.*]]: [[LWET:!lwe.lwe_ciphertext<.*>]]) -> [[LWET]]
func.func @boolean_gates(%arg0: !secret.secret<i1>) -> !secret.secret<i1> {
  // CHECK: [[VAL1:%.+]] = cggi.and [[ARG]], [[ARG]]
  // CHECK: [[VAL2:%.+]] = cggi.or [[VAL1]], [[ARG]]
  // CHECK: [[VAL3:%.+]] = cggi.nand [[VAL2]], [[VAL1]]
  // CHECK: [[VAL4:%.+]] = cggi.xor [[VAL3]], [[VAL2]]
  // CHECK: [[VAL5:%.+]] = cggi.xnor [[VAL4]], [[VAL3]]
  // CHECK: [[VAL6:%.+]] = cggi.nor [[VAL5]], [[VAL4]]
  %0 = secret.generic
      ins(%arg0:  !secret.secret<i1>) {
      ^bb0(%ARG0: i1) :
          %1 = comb.and %ARG0, %ARG0 : i1
          %2 = comb.or %1, %ARG0 : i1
          %3 = comb.nand %2, %1 : i1
          %4 = comb.xor %3, %2 : i1
          %5 = comb.xnor %4, %3 : i1
          %6 = comb.nor %5, %4 : i1
          secret.yield %6 : i1
      } -> (!secret.secret<i1>)
  // CHECK: return [[VAL6]] : [[LWET]]
  func.return %0 : !secret.secret<i1>
}

// -----

// CHECK-NOT: secret
// CHECK: @boolean_gates_partial_secret(
// CHECK-SAME:  [[ARG0:%.*]]: [[LWET:!lwe.lwe_ciphertext<.*>]], [[ARG1:%.*]]: i1) -> [[LWET]]
func.func @boolean_gates_partial_secret(%arg0: !secret.secret<i1>, %arg1 : i1) -> !secret.secret<i1> {
  // CHECK: [[ENC:%.+]] = lwe.encode [[ARG1]]
  // CHECK: [[LWE:%.+]] = lwe.trivial_encrypt [[ENC]]
  // CHECK: [[VAL1:%.+]] = cggi.and [[ARG0]], [[LWE]]
  %0 = secret.generic
      ins(%arg0, %arg1: !secret.secret<i1>, i1) {
      ^bb0(%ARG0: i1, %ARG1: i1) :
          %1 = comb.and %ARG0, %ARG1 : i1
          secret.yield %1 : i1
      } -> (!secret.secret<i1>)
  // CHECK: return [[VAL1]] : [[LWET]]
  func.return %0 : !secret.secret<i1>
}
