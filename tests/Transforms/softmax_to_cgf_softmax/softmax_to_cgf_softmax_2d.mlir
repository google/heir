// RUN: heir-opt --softmax-to-cgf-softmax %s | FileCheck %s

// CHECK: func @softmax_2d
// CHECK-SAME:  [[ARG0:%[a-zA-Z0-9_]+]]: tensor<2x8xf32>
func.func @softmax_2d(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // CHECK-NOT: math_ext.softmax

  // CHECK-DAG: [[CST_INV_N:%.+]] = arith.constant 1.250000e-01 : f32
  // CHECK-DAG: [[CST_HALF:%.+]] = arith.constant 5.000000e-01 : f32
  // CHECK-DAG: [[CST_LN_N:%.+]] = arith.constant 2.07944155 : f32

  // CHECK: [[SUM_BCAST:%.+]] = tensor_ext.broadcasted_reduce [[ARG0]] {dimension = 1 : i64, reduceOp = "arith.addf"} : tensor<2x8xf32>
  // CHECK: [[INV_N_SPLAT:%.+]] = tensor.splat [[CST_INV_N]] : tensor<2x8xf32>
  // CHECK: [[MU_BCAST:%.+]] = arith.mulf [[SUM_BCAST]], [[INV_N_SPLAT]] : tensor<2x8xf32>

  // CHECK: [[DIFF:%.+]] = arith.subf [[ARG0]], [[MU_BCAST]] : tensor<2x8xf32>
  // CHECK: [[DIFF_SQ:%.+]] = arith.mulf [[DIFF]], [[DIFF]] : tensor<2x8xf32>

  // CHECK: [[SUM_DIFF_SQ_BCAST:%.+]] = tensor_ext.broadcasted_reduce [[DIFF_SQ]] {dimension = 1 : i64, reduceOp = "arith.addf"} : tensor<2x8xf32>
  // CHECK: [[SIGMA_SQ_BCAST:%.+]] = arith.mulf [[SUM_DIFF_SQ_BCAST]], [[INV_N_SPLAT]] : tensor<2x8xf32>

  // CHECK: [[HALF_SPLAT:%.+]] = tensor.splat [[CST_HALF]] : tensor<2x8xf32>
  // CHECK: [[LN_N_SPLAT:%.+]] = tensor.splat [[CST_LN_N]] : tensor<2x8xf32>
  // CHECK: [[HALF_SIGMA_SQ_BCAST:%.+]] = arith.mulf [[SIGMA_SQ_BCAST]], [[HALF_SPLAT]] : tensor<2x8xf32>
  // CHECK: [[MU_HALF_SIGMA_SQ_BCAST:%.+]] = arith.addf [[MU_BCAST]], [[HALF_SIGMA_SQ_BCAST]] : tensor<2x8xf32>
  // CHECK: [[SHIFT_BCAST:%.+]] = arith.addf [[MU_HALF_SIGMA_SQ_BCAST]], [[LN_N_SPLAT]] : tensor<2x8xf32>

  // CHECK: [[SHIFTED_INPUT:%.+]] = arith.subf [[ARG0]], [[SHIFT_BCAST]] : tensor<2x8xf32>
  // CHECK: [[RESULT:%.+]] = math.exp [[SHIFTED_INPUT]] {domain_lower = -4.57944154{{[0-9]*}} : f64, domain_upper = 5.000000e-01 : f64} : tensor<2x8xf32>
  // CHECK: return [[RESULT]] : tensor<2x8xf32>

  %0 = math_ext.softmax %arg0 : tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
