// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

#poly = #polynomial.int_polynomial<-1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967296 : i64, polynomialModulus=#poly>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func @test_monomial_mul() -> !poly_ty {
  %deg = arith.constant 10 : index
  %five = arith.constant 5 : i32
  // CHECK: %[[MONOMIAL_DEG:.*]] = arith.constant 2 : index
  %monomial_degree = arith.constant 2 : index
  %0 = polynomial.monomial %five, %deg : (i32, index) -> !poly_ty
  // CHECK: %[[CONTAINER:.*]] = tensor.empty() : [[TENSOR_TYPE:tensor<1024xi32>]]
  // CHECK: %[[c1024:.*]] = arith.constant 1024 : index
  // CHECK: %[[SPLIT:.*]] = arith.subi %[[c1024]], %[[MONOMIAL_DEG]] : index

  // CHECK: %[[FIRST_HALF:.*]] = tensor.extract_slice
  // CHECK-SAME: [0] [%[[SPLIT]]] [1]

  // CHECK: %[[SECOND_HALF:.*]] = tensor.extract_slice
  // CHECK-SAME: [%[[SPLIT]]] [%[[MONOMIAL_DEG]]] [1]

  // CHECK: %[[FIRST_INSERT:.*]] = tensor.insert_slice %[[FIRST_HALF]] into %[[CONTAINER]]
  // CHECK-SAME: [%[[MONOMIAL_DEG]]] [%[[SPLIT]]] [1]

  // CHECK: tensor.insert_slice %[[SECOND_HALF]] into %[[FIRST_INSERT]]
  // CHECK-SAME: [0] [%[[MONOMIAL_DEG]]] [1]
  %1 = polynomial.monic_monomial_mul %0, %monomial_degree : (!poly_ty, index) -> !poly_ty
  return %1 : !poly_ty
}
