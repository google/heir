// RUN: heir-opt %s --lower-polynomial-eval='method=pscheb min-coefficient-threshold=1e-15' | FileCheck %s

!poly_ty = !polynomial.polynomial<ring=<coefficientType=f64>>
// This corresponds to a degree-10 ReLU approximation
#poly = #polynomial.typed_chebyshev_polynomial<[
      0.3181208873635371,    0.5,
      0.21259683947738586,   -5.867410479938111e-17,
      -0.042871530276047946, -1.8610028415987085e-17,
      0.018698658506314733,  -4.6179699483612605e-17,
      -0.010761656553922883, -3.366722265465115e-17,
      0.016587575708296047
]> : !poly_ty

// CHECK: @test_eval_for_paterson
func.func @test_eval_for_paterson() -> f64 {
// CHECK-DAG: [[CST:%.+]] = arith.constant 4.0{{0*}}e-01 : f64
// CHECK-DAG: [[CST_1:%.+]] = arith.constant 1.0
// CHECK-DAG: [[CST_0:%.+]] = arith.constant 0.32
// CHECK-NEXT: [[V0:%.+]] = arith.mulf [[CST_0]], [[CST_1]]
// CHECK-NEXT: [[CST_2:%.+]] = arith.constant 5.0{{0*}}e-01
// CHECK-NEXT: [[V1:%.+]] = arith.mulf [[CST_2]], [[CST]]
// CHECK-NEXT: [[V2:%.+]] = arith.addf [[V0]], [[V1]]
// CHECK-NEXT: [[CST_3:%.+]] = arith.constant 0.177
// CHECK-NEXT: [[V3:%.+]] = arith.mulf [[CST]], [[CST]]
// CHECK-NEXT: [[CST_4:%.+]] = arith.constant 2.0
// CHECK-NEXT: [[V4:%.+]] = arith.mulf [[V3]], [[CST_4]]
// CHECK-NEXT: [[V5:%.+]] = arith.subf [[V4]], [[CST_1]]
// CHECK-NEXT: [[V6:%.+]] = arith.mulf [[CST_3]], [[V5]]
// CHECK-NEXT: [[V7:%.+]] = arith.addf [[V2]], [[V6]]
// CHECK-NEXT: [[CST_5:%.+]] = arith.constant -0.042
// CHECK-NEXT: [[V8:%.+]] = arith.mulf [[CST_5]], [[CST_1]]
// CHECK-NEXT: [[CST_6:%.+]] = arith.constant 0.00422
// CHECK-NEXT: [[V9:%.+]] = arith.mulf [[CST_6]], [[V5]]
// CHECK-NEXT: [[V10:%.+]] = arith.addf [[V8]], [[V9]]
// CHECK-NEXT: [[V11:%.+]] = arith.mulf [[V5]], [[V5]]
// CHECK-NEXT: [[V12:%.+]] = arith.mulf [[V11]], [[CST_4]]
// CHECK-NEXT: [[V13:%.+]] = arith.subf [[V12]], [[CST_1]]
// CHECK-NEXT: [[V14:%.+]] = arith.mulf [[V10]], [[V13]]
// CHECK-NEXT: [[V15:%.+]] = arith.addf [[V7]], [[V14]]
// CHECK-NEXT: [[CST_7:%.+]] = arith.constant -0.0215
// CHECK-NEXT: [[V16:%.+]] = arith.mulf [[CST_7]], [[CST_1]]
// CHECK-NEXT: [[CST_8:%.+]] = arith.constant 0.0663
// CHECK-NEXT: [[V17:%.+]] = arith.mulf [[CST_8]], [[V5]]
// CHECK-NEXT: [[V18:%.+]] = arith.addf [[V16]], [[V17]]
// CHECK-NEXT: [[V19:%.+]] = arith.mulf [[V13]], [[V13]]
// CHECK-NEXT: [[V20:%.+]] = arith.mulf [[V18]], [[V19]]
// CHECK-NEXT: [[V21:%.+]] = arith.addf [[V15]], [[V20]]
// CHECK-NEXT: return [[V21]]
    %x = arith.constant 0.4 : f64
    %0 = polynomial.eval #poly, %x {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f64
    return %0 : f64
}
