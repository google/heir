// RUN: heir-translate --import-autohog %S/adder4.json | FileCheck %s

// CHECK-LABEL: func private @"4bit-4bit-adder"
// CHECK-SAME: (%[[arg0:.*]]: tensor<8x!lwe.lwe_ciphertext
// CHECK-SAME: -> tensor<5x!lwe.lwe_ciphertext

// CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[c6:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
// CHECK-DAG: %[[i3:.*]] = tensor.extract %[[arg0]][%[[c3]]]
// CHECK-DAG: %[[i4:.*]] = tensor.extract %[[arg0]][%[[c4]]]
// CHECK-DAG: %[[i6:.*]] = tensor.extract %[[arg0]][%[[c6]]]
// CHECK-DAG: %[[i7:.*]] = tensor.extract %[[arg0]][%[[c7]]]

// CHECK: %[[v0_outputs:.*]]:4 = cggi.multi_lut_lincomb %[[i7]], %[[i6]], %[[i4]], %[[i3]] {coefficients = array<i32: 1, 8, 4, 2>, lookup_tables = array<i32: 13260, 52224, 59552, 23130>}

// CHECK-DAG: %[[c0_1:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c2_1:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[c3_1:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[c4_1:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[c6_1:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[c7_1:.*]] = arith.constant 7 : index
// CHECK-DAG: %[[i0_1:.*]] = tensor.extract %[[arg0]][%[[c0_1]]]
// CHECK-DAG: %[[i2_1:.*]] = tensor.extract %[[arg0]][%[[c2_1]]]
// CHECK-DAG: %[[i3_1:.*]] = tensor.extract %[[arg0]][%[[c3_1]]]
// CHECK-DAG: %[[i4_1:.*]] = tensor.extract %[[arg0]][%[[c4_1]]]
// CHECK-DAG: %[[i6_1:.*]] = tensor.extract %[[arg0]][%[[c6_1]]]
// CHECK-DAG: %[[i7_1:.*]] = tensor.extract %[[arg0]][%[[c7_1]]]

// CHECK: %[[v1:.*]] = cggi.lut_lincomb %[[i3_1]], %[[i6_1]], %[[i4_1]], %[[i0_1]], %[[i2_1]], %[[i7_1]] {coefficients = array<i32: 1, 1, 2, 8, 8, 2>, lookup_table = 4132735 : i32}


// CHECK-DAG: %[[c0_2:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1_2:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c2_2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[c5_2:.*]] = arith.constant 5 : index
// CHECK-DAG: %[[i0_2:.*]] = tensor.extract %[[arg0]][%[[c0_2]]]
// CHECK-DAG: %[[i1_2:.*]] = tensor.extract %[[arg0]][%[[c1_2]]]
// CHECK-DAG: %[[i2_2:.*]] = tensor.extract %[[arg0]][%[[c2_2]]]
// CHECK-DAG: %[[i5_2:.*]] = tensor.extract %[[arg0]][%[[c5_2]]]

// CHECK: %[[v2_outputs:.*]]:3 = cggi.multi_lut_lincomb %[[i1_2]], %[[i0_2]], %[[i5_2]], %[[i2_2]], %[[v1]] {coefficients = array<i32: 2, 16, 8, 4, 1>, lookup_tables = array<i32: -969316711, -37429948, 252702960>}

// CHECK: %[[v3_outputs:.*]]:2 = cggi.multi_lut_lincomb %[[v2_outputs]]#1, %[[v0_outputs]]#2, %[[v0_outputs]]#0, %[[v0_outputs]]#1 {coefficients = array<i32: 1, 8, 4, 2>, lookup_tables = array<i32: 13260, 23130>}

// CHECK: %[[ret:.*]] = tensor.from_elements %[[v0_outputs]]#3, %[[v2_outputs]]#2, %[[v3_outputs]]#0, %[[v3_outputs]]#1, %[[v2_outputs]]#0 : tensor<5x!lwe.lwe_ciphertext
// CHECK: return %[[ret]]
