// RUN: heir-opt --mlir-print-local-scope --secret-distribute-generic --split-input-file --secret-to-cggi --cse %s | FileCheck %s

// CHECK-NOT: secret
// CHECK: @boolean_gates([[ARG:%.*]]: [[LWET:!lwe.lwe_ciphertext<.*message_type = i1.*>]]) -> [[LWET]]
func.func @boolean_gates(%arg0: !secret.secret<i1>) -> !secret.secret<i1> {
  // CHECK: [[VAL1:%.+]] = cggi.and [[ARG]], [[ARG]] : [[LWET]]
  // CHECK: [[VAL2:%.+]] = cggi.or [[VAL1]], [[ARG]] : [[LWET]]
  // CHECK: [[VAL3:%.+]] = cggi.nand [[VAL2]], [[VAL1]] : [[LWET]]
  // CHECK: [[VAL4:%.+]] = cggi.xor [[VAL3]], [[VAL2]] : [[LWET]]
  // CHECK: [[VAL5:%.+]] = cggi.xnor [[VAL4]], [[VAL3]] : [[LWET]]
  // CHECK: [[VAL6:%.+]] = cggi.nor [[VAL5]], [[VAL4]] : [[LWET]]
  %0 = secret.generic
      (%arg0:  !secret.secret<i1>) {
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
  // CHECK: [[VAL1:%.+]] = cggi.and [[ARG0]], [[LWE]] : [[LWET]]
  %0 = secret.generic
      (%arg0: !secret.secret<i1>, %arg1: i1) {
      ^bb0(%ARG0: i1, %ARG1: i1) :
          %1 = comb.and %ARG0, %ARG1 : i1
          secret.yield %1 : i1
      } -> (!secret.secret<i1>)
  // CHECK: return [[VAL1]] : [[LWET]]
  func.return %0 : !secret.secret<i1>
}
