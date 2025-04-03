// RUN: heir-opt --mlir-print-local-scope --polynomial-to-mod-arith %s | FileCheck %s

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<4294967296:i64>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl_2048>
!poly_ty = !polynomial.polynomial<ring=#ring>

!int_coeff_ty = i32
#int_ring = #polynomial.ring<coefficientType=!int_coeff_ty, polynomialModulus=#cycl_2048>
!int_poly_ty = !polynomial.polynomial<ring=#int_ring>

// CHECK: @lower_mod_arith_coeffs
func.func @lower_mod_arith_coeffs() -> !poly_ty {
  // CHECK-DAG: [[X:%.+]] = arith.constant dense<"0x01
  // CHECK-DAG: [[Y:%.+]] = arith.constant dense<"0x02
  // CHECK-DAG: [[Xmod:%.+]] = mod_arith.encapsulate [[X]] : tensor<1024xi64> -> tensor<1024x!mod_arith.int<4294967296 : i64>>
  // CHECK-DAG: [[Ymod:%.+]] = mod_arith.encapsulate [[Y]] : tensor<1024xi64> -> tensor<1024x!mod_arith.int<4294967296 : i64>>
  // CHECK: [[ADD:%.+]] = mod_arith.add [[Xmod]], [[Ymod]]
  // CHECK: return [[ADD]] : tensor<1024x!mod_arith.int<4294967296 : i64>>
  %0 = polynomial.constant int<1 + x**10> : !poly_ty
  %1 = polynomial.constant int<2 + 3x**9> : !poly_ty
  %poly2 = polynomial.add %0, %1 : !polynomial.polynomial<ring=#ring>
  return %poly2 : !polynomial.polynomial<ring=#ring>
}

// CHECK: @lower_int_coeffs
func.func @lower_int_coeffs() -> !int_poly_ty {
  // CHECK-DAG: [[X:%.+]] = arith.constant dense<"0x01
  // CHECK-DAG: [[Y:%.+]] = arith.constant dense<"0x02
  // CHECK: [[ADD:%.+]] = arith.addi [[X]], [[Y]]
  // CHECK: return [[ADD]] : tensor<1024xi32>
  %0 = polynomial.constant int<1 + x**10> : !int_poly_ty
  %1 = polynomial.constant int<2 + 3x**9> : !int_poly_ty
  %poly2 = polynomial.add %0, %1 : !int_poly_ty
  return %poly2 : !int_poly_ty
}

// CHECK: @test_lower_add_tensor
func.func @test_lower_add_tensor() -> tensor<2x!poly_ty> {
  // 2 + 2x + 2x^2 + ... + 2x^{1023}
  // CHECK-DAG: [[A:%.+]] = arith.constant dense<2> : [[T:tensor<1024xi64>]]
  // CHECK-DAG: [[B:%.+]] = arith.constant dense<3> : [[T]]
  // CHECK-DAG: [[C:%.+]] = arith.constant dense<4> : [[T]]
  // CHECK-DAG: [[D:%.+]] = arith.constant dense<5> : [[T]]
  // CHECK-DAG: [[Amod:%.+]] = mod_arith.encapsulate [[A]] : [[T]] -> [[Tmod:.*]]
  // CHECK-DAG: [[Bmod:%.+]] = mod_arith.encapsulate [[B]]
  // CHECK-DAG: [[Cmod:%.+]] = mod_arith.encapsulate [[C]]
  // CHECK-DAG: [[Dmod:%.+]] = mod_arith.encapsulate [[D]]
  %coeffsA_ = arith.constant dense<2> : tensor<1024xi64>
  %coeffsB_ = arith.constant dense<3> : tensor<1024xi64>
  %coeffsC_ = arith.constant dense<4> : tensor<1024xi64>
  %coeffsD_ = arith.constant dense<5> : tensor<1024xi64>
  %coeffsA = mod_arith.encapsulate %coeffsA_ : tensor<1024xi64> -> tensor<1024x!coeff_ty>
  %coeffsB = mod_arith.encapsulate %coeffsB_ : tensor<1024xi64> -> tensor<1024x!coeff_ty>
  %coeffsC = mod_arith.encapsulate %coeffsC_ : tensor<1024xi64> -> tensor<1024x!coeff_ty>
  %coeffsD = mod_arith.encapsulate %coeffsD_ : tensor<1024xi64> -> tensor<1024x!coeff_ty>
  %polyA = polynomial.from_tensor %coeffsA : tensor<1024x!coeff_ty> -> !poly_ty
  %polyB = polynomial.from_tensor %coeffsB : tensor<1024x!coeff_ty> -> !poly_ty
  %polyC = polynomial.from_tensor %coeffsC : tensor<1024x!coeff_ty> -> !poly_ty
  %polyD = polynomial.from_tensor %coeffsD : tensor<1024x!coeff_ty> -> !poly_ty
  %tensor1 = tensor.from_elements %polyA, %polyB : tensor<2x!polynomial.polynomial<ring=#ring>>
  %tensor2 = tensor.from_elements %polyC, %polyD : tensor<2x!polynomial.polynomial<ring=#ring>>
  // CHECK: [[S1:%.+]] = arith.constant dense<[1, 1024]> : [[TI:tensor<2xindex>]]
  // CHECK: [[T1:%.+]] = tensor.reshape [[Amod]]([[S1]]) : ([[Tmod]], [[TI]]) -> [[TEX:tensor<1x1024x!mod_arith.*>]]
  // CHECK: [[S2:%.+]] = arith.constant dense<[1, 1024]> : [[TI]]
  // CHECK: [[T2:%.+]] = tensor.reshape [[Bmod]]([[S2]]) : ([[Tmod]], [[TI]]) -> [[TEX]]
  // CHECK: [[C1:%.+]] = tensor.concat dim(0) [[T1]], [[T2]] : ([[TEX]], [[TEX]]) -> [[TT:tensor<2x1024x!mod_arith.*>]]
  // CHECK: [[S3:%.+]] = arith.constant dense<[1, 1024]> : [[TI]]
  // CHECK: [[T3:%.+]] = tensor.reshape [[Cmod]]([[S3]]) : ([[Tmod]], [[TI]]) -> [[TEX]]
  // CHECK: [[S4:%.+]] = arith.constant dense<[1, 1024]> : [[TI]]
  // CHECK: [[T4:%.+]] = tensor.reshape [[Dmod]]([[S4]]) : ([[Tmod]], [[TI]]) -> [[TEX]]
  // CHECK: [[C2:%.+]] = tensor.concat dim(0) [[T3]], [[T4]] : ([[TEX]], [[TEX]]) -> [[TT:tensor<2x1024x!mod_arith.*>]]
  // CHECK-NOT: polynomial.from_tensor
  // CHECK-NOT: tensor.from_elements
  %tensor3 = affine.for %i = 0 to 2 iter_args(%t0 = %tensor1) ->  tensor<2x!polynomial.polynomial<ring=#ring>> {
      // CHECK: [[FOR:%.]] = affine.for [[I:%.+]] = 0 to 2 iter_args([[T0:%.+]] = [[C1]]) -> ([[TT]]) {
      %a = tensor.extract %tensor1[%i] :  tensor<2x!polynomial.polynomial<ring=#ring>>
      %b = tensor.extract %tensor2[%i] :  tensor<2x!polynomial.polynomial<ring=#ring>>
      // CHECK: [[AA:%.+]] = tensor.extract_slice [[C1]][[[I]], 0] [1, 1024] [1, 1] : [[TT]]
      // CHECK: [[BB:%.+]] = tensor.extract_slice [[C2]][[[I]], 0] [1, 1024] [1, 1] : [[TT]]
      // CHECK-NOT: tensor.extract %
      %s = polynomial.add %a, %b : !polynomial.polynomial<ring=#ring>
      // CHECK: [[SUM:%.+]] = mod_arith.add [[AA]], [[BB]] : [[Tmod]]
      // CHECK-NOT: polynomial.add
      %t = tensor.insert %s into %t0[%i] :  tensor<2x!polynomial.polynomial<ring=#ring>>
      // CHECK: [[INS:%.+]] = tensor.insert_slice [[SUM]] into [[T0]][[[I]], 0] [1, 1024] [1, 1] : [[Tmod]] into [[TT]]
      // CHECK-NOT: tensor.insert %
      affine.yield %t :  tensor<2x!polynomial.polynomial<ring=#ring>>
      // CHECK: affine.yield [[INS]] : [[TT]]
    }
  return %tensor3 :  tensor<2x!polynomial.polynomial<ring=#ring>>
  // CHECK: return [[FOR]] : [[TT]]
}
