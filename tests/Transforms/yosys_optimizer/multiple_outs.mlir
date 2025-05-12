// RUN: heir-opt --yosys-optimizer --canonicalize --symbol-dce %s | FileCheck %s

// CHECK: func.func @multi_output(%[[arg0:.*]]: !secret.secret<i16>)
func.func @multi_output(%arg0: !secret.secret<i16>) -> (!secret.secret<i16>, !secret.secret<i16>) {
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[false:.*]] = arith.constant false
  // CHECK: %[[v0:.*]] = secret.cast %[[arg0]] : !secret.secret<i16> to !secret.secret<tensor<16xi1>>
  //
  // CHECK: %[[v1:.*]]:2 = secret.generic(%[[v0]]: !secret.secret<tensor<16xi1>>) {
  // CHECK-NEXT: ^body(%[[input0:.*]]: tensor<16xi1>):
  // CHECK-COUNT-15: tensor.extract %[[input0]]
  // CHECK: %[[alloc:.*]] = tensor.from_elements %[[false]]
  // CHECK-SAME: tensor<16xi1>
  // CHECK-NEXT:   secret.yield %[[alloc]], %[[alloc]]
  //
  // CHECK-DAG: %[[v2:.*]] = secret.cast %[[v1]]#0
  // CHECK-DAG: %[[v3:.*]] = secret.cast %[[v1]]#1
  // CHECK: return %[[v2]], %[[v3]]
  %c2_i16 = arith.constant 2 : i16
  %0:2 = secret.generic(%arg0 : !secret.secret<i16>) {
  ^body(%input0: i16):
    %1 = arith.muli %input0, %c2_i16 : i16
    secret.yield %1, %1 : i16, i16
  } -> (!secret.secret<i16>, !secret.secret<i16>)
  return %0#0, %0#1 : !secret.secret<i16>, !secret.secret<i16>
}
