// RUN: heir-opt --straight-line-vectorize="dialect=cggi" %s

#key = #lwe.key<slot_index = 0>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!pt_ty = !lwe.lwe_plaintext<plaintext_space = #pspace>
!ct_ty = !lwe.lwe_ciphertext<plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK: arith_and_cggi
// CHECK-COUNT-1: cggi.and
// CHECK-COUNT-2: arith.andi
// CHECK-NOT: cggi.and
// CHECK: return
func.func @arith_and_cggi(%arg0: tensor<4x!ct_ty>, %arg1 : i8, %arg2 : i8) -> (tensor<2x!ct_ty>, tensor<2xi8>) {
  %c0_i8 = arith.constant 0 : i8
  %c1_i8 = arith.constant 1 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %extracted = tensor.extract %arg0[%c0] : tensor<4x!ct_ty>
  %extracted_0 = tensor.extract %arg0[%c1] : tensor<4x!ct_ty>
  %extracted_1 = tensor.extract %arg0[%c2] : tensor<4x!ct_ty>
  %extracted_2 = tensor.extract %arg0[%c3] : tensor<4x!ct_ty>
  %0 = cggi.and %extracted, %extracted_0 : !ct_ty
  %1 = cggi.and %extracted_1, %extracted_2 : !ct_ty
  %from_elements = tensor.from_elements %0, %1 : tensor<2x!ct_ty>
  %2 = arith.andi %arg1, %c0_i8 : i8
  %3 = arith.andi %arg2, %c1_i8 : i8
  %from_elements_0 = tensor.from_elements %2, %3 : tensor<2xi8>
  return %from_elements, %from_elements_0 : tensor<2x!ct_ty>, tensor<2xi8>
}
