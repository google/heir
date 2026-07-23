// RUN: heir-opt --softmax-to-cgf-softmax %s | FileCheck %s

// CHECK: func @softmax_simple
// CHECK-SAME:  [[ARG0:%[a-zA-Z0-9_]+]]: tensor<8xf32>
func.func @softmax_simple(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NOT: math_ext.softmax

  // CHECK-DAG: [[CST_INV_N:%.+]] = arith.constant 1.250000e-01 : f32
  // CHECK-DAG: [[CST_HALF:%.+]] = arith.constant 5.000000e-01 : f32
  // CHECK-DAG: [[CST_LN_N:%.+]] = arith.constant 2.07944155 : f32

  // CHECK: [[FILL_0:%.+]] = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[REDUCE_0:%.+]] = linalg.reduce ins([[ARG0]] : tensor<8xf32>) outs([[FILL_0]] : tensor<f32>) dimensions = [0]
  // CHECK:   ([[IN_0:%.+]]: f32, [[ACC_0:%.+]]: f32) {
  // CHECK:     [[ADD_0:%.+]] = arith.addf [[IN_0]], [[ACC_0]] : f32
  // CHECK:     linalg.yield [[ADD_0]] : f32
  // CHECK: }

  // CHECK: [[INV_N_SPLAT:%.+]] = tensor.splat [[CST_INV_N]] : tensor<f32>
  // CHECK: [[MU:%.+]] = arith.mulf [[REDUCE_0]], [[INV_N_SPLAT]] : tensor<f32>

  // CHECK: [[EMPTY_1D:%.+]] = tensor.empty() : tensor<8xf32>
  // CHECK: [[MU_BCAST:%.+]] = linalg.broadcast ins([[MU]] : tensor<f32>) outs([[EMPTY_1D]] : tensor<8xf32>) dimensions = [0]
  // CHECK: [[DIFF:%.+]] = arith.subf [[ARG0]], [[MU_BCAST]] : tensor<8xf32>
  // CHECK: [[DIFF_SQ:%.+]] = arith.mulf [[DIFF]], [[DIFF]] : tensor<8xf32>

  // CHECK: [[FILL_1:%.+]] = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[REDUCE_1:%.+]] = linalg.reduce ins([[DIFF_SQ]] : tensor<8xf32>) outs([[FILL_1]] : tensor<f32>) dimensions = [0]
  // CHECK:   ([[IN_2:%.+]]: f32, [[ACC_2:%.+]]: f32) {
  // CHECK:     [[ADD_1:%.+]] = arith.addf [[IN_2]], [[ACC_2]] : f32
  // CHECK:     linalg.yield [[ADD_1]] : f32
  // CHECK: }

  // CHECK: [[SIGMA_SQ:%.+]] = arith.mulf [[REDUCE_1]], [[INV_N_SPLAT]] : tensor<f32>
  // CHECK: [[HALF_SPLAT:%.+]] = tensor.splat [[CST_HALF]] : tensor<f32>
  // CHECK: [[LN_N_SPLAT:%.+]] = tensor.splat [[CST_LN_N]] : tensor<f32>
  // CHECK: [[HALF_SIGMA_SQ:%.+]] = arith.mulf [[SIGMA_SQ]], [[HALF_SPLAT]] : tensor<f32>
  // CHECK: [[MU_HALF_SIGMA_SQ:%.+]] = arith.addf [[MU]], [[HALF_SIGMA_SQ]] : tensor<f32>
  // CHECK: [[SHIFT:%.+]] = arith.addf [[MU_HALF_SIGMA_SQ]], [[LN_N_SPLAT]] : tensor<f32>

  // CHECK: [[SHIFT_BCAST:%.+]] = linalg.broadcast ins([[SHIFT]] : tensor<f32>) outs([[EMPTY_1D]] : tensor<8xf32>) dimensions = [0]
  // CHECK: [[SHIFTED_INPUT:%.+]] = arith.subf [[ARG0]], [[SHIFT_BCAST]] : tensor<8xf32>
  // CHECK: [[RESULT:%.+]] = math.exp [[SHIFTED_INPUT]] {domain_lower = -4.57944154{{[0-9]*}} : f64, domain_upper = 5.000000e-01 : f64} : tensor<8xf32>
  // CHECK: return [[RESULT]] : tensor<8xf32>

  %0 = math_ext.softmax %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
