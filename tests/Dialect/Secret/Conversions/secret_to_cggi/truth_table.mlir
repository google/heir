// RUN: heir-opt --secret-distribute-generic --canonicalize --split-input-file --secret-to-cggi --cse %s | FileCheck %s

// CHECK: ![[ct_ty:.*]] = !lwe.lwe_ciphertext

// CHECK-NOT: secret
// CHECK: @truth_table_all_secret([[ARG:%.*]]: ![[ct_ty]]) -> ![[ct_ty]]
func.func @truth_table_all_secret(%arg0: !secret.secret<i1>) -> !secret.secret<i1> {
  // CHECK: [[VAL:%.+]] = cggi.lut3 [[ARG]], [[ARG]], [[ARG]]
  %0 = secret.generic
      (%arg0:  !secret.secret<i1>) {
      ^bb0(%ARG0: i1) :
          %1 = comb.truth_table %ARG0, %ARG0, %ARG0 -> 6 : ui8
          secret.yield %1 : i1
      } -> (!secret.secret<i1>)
  // CHECK: return [[VAL]] : ![[ct_ty]]
  func.return %0 : !secret.secret<i1>
}

// -----

// CHECK: ![[ct_ty:.*]] = !lwe.lwe_ciphertext

// CHECK-NOT: secret
// CHECK: @truth_table_partial_secret([[ARG:%.*]]: ![[ct_ty]]) -> ![[ct_ty]]
func.func @truth_table_partial_secret(%arg0: !secret.secret<i1>) -> !secret.secret<i1> {
  // CHECK: [[FALSE:%.+]] = arith.constant false
  %false = arith.constant false
  // CHECK: [[TRUE:%.+]] = arith.constant true
  %true = arith.constant true
  // CHECK: [[ENCFALSE:%.+]] = lwe.encode [[FALSE]]
  // CHECK: [[LWEFALSE:%.+]] = lwe.trivial_encrypt [[ENCFALSE]]
  // CHECK: [[ENCTRUE:%.+]] = lwe.encode [[TRUE]]
  // CHECK: [[LWETRUE:%.+]] = lwe.trivial_encrypt [[ENCTRUE]]
  // CHECK: [[VAL1:%.+]] = cggi.lut3 [[LWEFALSE]], [[LWETRUE]], [[ARG]]
  // CHECK: [[VAL2:%.+]] = cggi.lut3 [[LWEFALSE]], [[LWETRUE]], [[VAL1]]
  %0 = secret.generic
      (%false: i1, %true: i1, %arg0: !secret.secret<i1>) {
      ^bb0(%FALSE: i1, %TRUE: i1, %ARG0: i1) :
          %1 = comb.truth_table %FALSE, %TRUE, %ARG0 -> 6 : ui8
          %2 = comb.truth_table %FALSE, %TRUE, %1 -> 2 : ui8
          secret.yield %2 : i1
      } -> (!secret.secret<i1>)
  // CHECK: return [[VAL2]] : ![[ct_ty]]
  func.return %0 : !secret.secret<i1>
}

// -----

// CHECK: ![[ct_ty:.*]] = !lwe.lwe_ciphertext

// CHECK-NOT: secret
// CHECK: @truth_table_no_secret([[ARG:%.*]]: ![[ct_ty]], [[BOOL1:%.*]]: i1, [[BOOL2:%.*]]: i1) -> ![[ct_ty]]
func.func @truth_table_no_secret(%arg0: !secret.secret<i1>, %bool1: i1, %bool2: i1) -> !secret.secret<i1> {
  %false = arith.constant false
  // CHECK: [[TRUE:%.+]] = arith.constant true
  // CHECK: [[FALSE:%.+]] = arith.constant false
  %true = arith.constant true
  // CHECK-COUNT-2: arith.select
  // CHECK: [[VAL:%.+]] = arith.select
  // CHECK: [[ENCTRUE:%.+]] = lwe.encode [[TRUE]]
  // CHECK: [[LWETRUE:%.+]] = lwe.trivial_encrypt [[ENCTRUE]]
  // CHECK: [[VALENCODE:%.+]] = lwe.encode [[VAL]]
  // CHECK: [[VAL1:%.+]] = lwe.trivial_encrypt [[VALENCODE]]
  // CHECK: [[VAL2:%.+]] = cggi.lut3 [[ARG]], [[LWETRUE]], [[VAL1]]
  %0 = secret.generic
      (%false: i1, %true: i1, %bool1: i1, %bool2: i1, %arg0: !secret.secret<i1>) {
      ^bb0(%FALSE: i1, %TRUE: i1, %BOOL1: i1, %BOOL2: i1, %ARG0: i1) :
          %1 = comb.truth_table %BOOL1, %BOOL2, %BOOL1 -> 6 : ui8
          %2 = comb.truth_table %ARG0, %TRUE, %1 -> 2 : ui8
          secret.yield %2 : i1
      } -> (!secret.secret<i1>)
  // CHECK: return [[VAL2]] : ![[ct_ty]]
  func.return %0 : !secret.secret<i1>
}
