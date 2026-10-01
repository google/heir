// RUN: heir-opt --lwe-to-lattigo %s | FileCheck %s

!Z65537_i64 = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 1>
#ring_Z65537_i64_1_x32 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**32>>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_Z65537_i64_1_x32, encoding = #full_crt_packing_encoding>>

// CHECK-DAG: ![[ENC:.*]] = !lattigo.bgv.encoder
// CHECK-DAG: ![[EVAL:.*]] = !lattigo.bgv.evaluator
// CHECK-DAG: ![[PARAM:.*]] = !lattigo.bgv.parameter
// CHECK-DAG: ![[PT:.*]] = !lattigo.rlwe.plaintext
// CHECK-DAG: ![[LT:.*]] = !lattigo.ckks.linear_transformation

// CHECK: func.func @test_preprocessing(%{{.*}}: ![[EVAL]], %{{.*}}: ![[PARAM]], %{{.*}}: ![[ENC]], %[[ARG0:.*]]: ![[PT]]) -> ![[PT]]
// CHECK: %[[storage:.*]] = preprocessing.empty : <![[PT]]>
// CHECK: preprocessing.store %[[ARG0]], %[[storage]][] site 0<![[PT]]> : ![[PT]], <![[PT]]>
// CHECK: %[[res:.*]] = preprocessing.load %[[storage]][] site 0<![[PT]]> : <![[PT]]>, ![[PT]]
// CHECK: return %[[res]] : ![[PT]]
module attributes {scheme.bgv} {
  func.func @test_preprocessing(%arg0: !pt) -> !pt {
    %storage = preprocessing.empty : !preprocessing.storage<!pt>
    preprocessing.store %arg0, %storage[] site 0 <!pt> : !pt, !preprocessing.storage<!pt>
    %res = preprocessing.load %storage[] site 0 <!pt> : !preprocessing.storage<!pt>, !pt
    return %res : !pt
  }

  // CHECK: func.func @test_preprocessing_multi(%{{.*}}: ![[EVAL]], %{{.*}}: ![[PARAM]], %{{.*}}: ![[ENC]], %[[ARG0:.*]]: ![[PT]], %[[ARG1:.*]]: ![[LT]], %[[ARG2:.*]]: ![[LT]]) -> (![[LT]], ![[LT]])
  // CHECK: %[[storage:.*]] = preprocessing.empty : <![[PT]], ![[LT]]>
  // CHECK: preprocessing.store %[[ARG0]], %[[storage]][] site 0<![[PT]]> : ![[PT]], <![[PT]], ![[LT]]>
  // CHECK: preprocessing.store %[[ARG1]], %[[storage]][] site 1<![[LT]]> : ![[LT]], <![[PT]], ![[LT]]>
  // CHECK: preprocessing.store %[[ARG2]], %[[storage]][] site 2<![[LT]]> : ![[LT]], <![[PT]], ![[LT]]>
  // CHECK: %[[res1:.*]] = preprocessing.load %[[storage]][] site 1<![[LT]]> : <![[PT]], ![[LT]]>, ![[LT]]
  // CHECK: %[[res2:.*]] = preprocessing.load %[[storage]][] site 2<![[LT]]> : <![[PT]], ![[LT]]>, ![[LT]]
  // CHECK: return %[[res1]], %[[res2]] : ![[LT]], ![[LT]]
  func.func @test_preprocessing_multi(
      %arg0: !pt,
      %arg1: !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>,
      %arg2: !kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>)
      -> (!kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>,
          !kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>) {
    %storage = preprocessing.empty : !preprocessing.storage<!pt, !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>, !kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>>
    preprocessing.store %arg0, %storage[] site 0 <!pt> : !pt, !preprocessing.storage<!pt, !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>, !kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>>
    preprocessing.store %arg1, %storage[] site 1 <!kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>> : !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>, !preprocessing.storage<!pt, !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>, !kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>>
    preprocessing.store %arg2, %storage[] site 2 <!kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>> : !kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>, !preprocessing.storage<!pt, !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>, !kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>>
    %res1 = preprocessing.load %storage[] site 1 <!kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>> : !preprocessing.storage<!pt, !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>, !kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>>, !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
    %res2 = preprocessing.load %storage[] site 2 <!kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>> : !preprocessing.storage<!pt, !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>, !kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>>, !kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>
    return %res1, %res2 : !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>, !kernel.prepared_linear_transform<level = 1, slots = 4, log_bsgs_ratio = 0>
  }
}
