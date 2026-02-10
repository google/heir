// RUN: heir-opt --cggi-to-jaxite %s | FileCheck %s

#key = #lwe.key<slot_index = 0>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!pt_ty = !lwe.lwe_plaintext<plaintext_space = #pspace>
!ct_ty = !lwe.lwe_ciphertext<plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK: test_add_one_lut3
// CHECK-COUNT-1: jaxite.pmap_lut3
func.func @test_add_one_lut3(%arg0: tensor<8x!ct_ty>) {
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
    %extracted_0 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
    %extracted_1 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>
    %extracted_2 = tensor.extract %arg0[%c3] : tensor<8x!ct_ty>
    %extracted_3 = tensor.extract %arg0[%c4] : tensor<8x!ct_ty>
    %extracted_4 = tensor.extract %arg0[%c5] : tensor<8x!ct_ty>
    %extracted_5 = tensor.extract %arg0[%c6] : tensor<8x!ct_ty>
    %extracted_6 = tensor.extract %arg0[%c7] : tensor<8x!ct_ty>

    %from_elements = tensor.from_elements %extracted, %extracted_1, %extracted_2, %extracted_3 : tensor<4x!ct_ty>
    %from_elements_7 = tensor.from_elements %extracted_3, %extracted_4, %extracted_5, %extracted_6 : tensor<4x!ct_ty>
    %from_elements_8 = tensor.from_elements %extracted_4, %extracted_5, %extracted_6, %extracted_2 : tensor<4x!ct_ty>
    %7 = cggi.packed_lut3 %from_elements, %from_elements_7, %from_elements_8 {lookup_tables = [6 : ui8, 120 : ui8, 6 : ui8, 120 : ui8]} : (tensor<4x!ct_ty>, tensor<4x!ct_ty>, tensor<4x!ct_ty>) -> tensor<4x!ct_ty>

    %c0_9 = arith.constant 0 : index
    %extracted_10 = tensor.extract %7[%c0_9] : tensor<4x!ct_ty>
    %c1_11 = arith.constant 1 : index
    %extracted_12 = tensor.extract %7[%c1_11] : tensor<4x!ct_ty>
    %c2_13 = arith.constant 2 : index
    %extracted_14 = tensor.extract %7[%c2_13] : tensor<4x!ct_ty>
    %c3_15 = arith.constant 3 : index
    %extracted_16 = tensor.extract %7[%c3_15] : tensor<4x!ct_ty>
    return
}
