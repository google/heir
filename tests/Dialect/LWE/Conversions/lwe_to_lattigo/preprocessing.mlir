// RUN: heir-opt --lwe-to-lattigo %s | FileCheck %s

!Z65537_i64 = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 1>
#ring_Z65537_i64_1_x32 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**32>>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_Z65537_i64_1_x32, encoding = #full_crt_packing_encoding>>

// CHECK: ![[ENC:.*]] = !lattigo.bgv.encoder
// CHECK: ![[EVAL:.*]] = !lattigo.bgv.evaluator
// CHECK: ![[PARAM:.*]] = !lattigo.bgv.parameter
// CHECK: ![[PT:.*]] = !lattigo.rlwe.plaintext

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
}
