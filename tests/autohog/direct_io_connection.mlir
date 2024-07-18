// RUN: heir-translate --import-autohog %S/direct_io_connection.json | FileCheck %s

// CHECK-LABEL: func.func private @direct_io_connection(
// CHECK-SAME: %[[arg0:.*]]: tensor<2x!lwe.lwe_ciphertext
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[v0:.*]] = tensor.extract %[[arg0]][%[[c0]]]
// CHECK-DAG: %[[v1:.*]] = tensor.extract %[[arg0]][%[[c1]]]
// CHECK: %[[v2:.*]] = cggi.and
// CHECK: tensor.from_elements %[[v2]], %[[v1]]
