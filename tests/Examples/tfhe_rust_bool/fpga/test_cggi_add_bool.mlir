// This test ensures the testing harness is working properly with minimal codegen.
// This function evaluates the addition of two 8bit unsigned integers using a boolean circuit.
// The bool circuit first consists of a half adder, then seven full adders

// heir-opt --straight-line-vectorize --cggi-to-tfhe-rust-bool -cse -remove-dead-values %s | heir-translate --emit-tfhe-rust-bool-packed > %S/src/fn_under_test.rs
// cargo run --release --manifest-path %S/Cargo.toml -- 1 1 | FileCheck %s

#constant_coefficient_encoding = #lwe.constant_coefficient_encoding<scaling_factor = 536870912>
#key = #lwe.key<>
#ring_i32_1 = #polynomial.ring<coefficientType = i32, polynomialModulus = <1>>
#ring_i3_1 = #polynomial.ring<coefficientType = i3, polynomialModulus = <1>>
#ciphertext_space__D742 = #lwe.ciphertext_space<ring = #ring_i32_1, encryption_type = msb, size = 742>
!ct__D742 = !lwe.lwe_ciphertext<application_data = <message_type = i1, overflow = #lwe.preserve_overflow<>>, plaintext_space = <ring = #ring_i3_1, encoding = #constant_coefficient_encoding>, ciphertext_space = #ciphertext_space__D742, key = #key>

module attributes {scheme.cggi} {
  func.func @fn_under_test(%arg0: tensor<8x!ct__D742>, %arg1: tensor<8x!ct__D742>) -> tensor<8x!ct__D742> {
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg0[%c0] : tensor<8x!ct__D742>
    %extracted_0 = tensor.extract %arg1[%c0] : tensor<8x!ct__D742>
    %ct = cggi.and %extracted, %extracted_0 : !ct__D742
    %extracted_1 = tensor.extract %arg0[%c1] : tensor<8x!ct__D742>
    %extracted_2 = tensor.extract %arg1[%c1] : tensor<8x!ct__D742>
    %ct_3 = cggi.nand %extracted_1, %extracted_2 : !ct__D742
    %ct_4 = cggi.xor %extracted_1, %extracted_2 : !ct__D742
    %ct_5 = cggi.nand %ct, %ct_4 : !ct__D742
    %ct_6 = cggi.xor %ct, %ct_4 : !ct__D742
    %ct_7 = cggi.nand %ct_3, %ct_5 : !ct__D742
    %extracted_8 = tensor.extract %arg0[%c2] : tensor<8x!ct__D742>
    %extracted_9 = tensor.extract %arg1[%c2] : tensor<8x!ct__D742>
    %ct_10 = cggi.nand %extracted_8, %extracted_9 : !ct__D742
    %ct_11 = cggi.xor %extracted_8, %extracted_9 : !ct__D742
    %ct_12 = cggi.nand %ct_7, %ct_11 : !ct__D742
    %ct_13 = cggi.xor %ct_7, %ct_11 : !ct__D742
    %ct_14 = cggi.nand %ct_10, %ct_12 : !ct__D742
    %extracted_15 = tensor.extract %arg0[%c3] : tensor<8x!ct__D742>
    %extracted_16 = tensor.extract %arg1[%c3] : tensor<8x!ct__D742>
    %ct_17 = cggi.nand %extracted_15, %extracted_16 : !ct__D742
    %ct_18 = cggi.xor %extracted_15, %extracted_16 : !ct__D742
    %ct_19 = cggi.nand %ct_14, %ct_18 : !ct__D742
    %ct_20 = cggi.xor %ct_14, %ct_18 : !ct__D742
    %ct_21 = cggi.nand %ct_17, %ct_19 : !ct__D742
    %extracted_22 = tensor.extract %arg0[%c4] : tensor<8x!ct__D742>
    %extracted_23 = tensor.extract %arg1[%c4] : tensor<8x!ct__D742>
    %ct_24 = cggi.nand %extracted_22, %extracted_23 : !ct__D742
    %ct_25 = cggi.xor %extracted_22, %extracted_23 : !ct__D742
    %ct_26 = cggi.nand %ct_21, %ct_25 : !ct__D742
    %ct_27 = cggi.xor %ct_21, %ct_25 : !ct__D742
    %ct_28 = cggi.nand %ct_24, %ct_26 : !ct__D742
    %extracted_29 = tensor.extract %arg0[%c5] : tensor<8x!ct__D742>
    %extracted_30 = tensor.extract %arg1[%c5] : tensor<8x!ct__D742>
    %ct_31 = cggi.nand %extracted_29, %extracted_30 : !ct__D742
    %ct_32 = cggi.xor %extracted_29, %extracted_30 : !ct__D742
    %ct_33 = cggi.nand %ct_28, %ct_32 : !ct__D742
    %ct_34 = cggi.xor %ct_28, %ct_32 : !ct__D742
    %ct_35 = cggi.nand %ct_31, %ct_33 : !ct__D742
    %extracted_36 = tensor.extract %arg0[%c6] : tensor<8x!ct__D742>
    %extracted_37 = tensor.extract %arg1[%c6] : tensor<8x!ct__D742>
    %ct_38 = cggi.nand %extracted_36, %extracted_37 : !ct__D742
    %ct_39 = cggi.xor %extracted_36, %extracted_37 : !ct__D742
    %ct_40 = cggi.nand %ct_35, %ct_39 : !ct__D742
    %ct_41 = cggi.xor %ct_35, %ct_39 : !ct__D742
    %ct_42 = cggi.nand %ct_38, %ct_40 : !ct__D742
    %extracted_43 = tensor.extract %arg0[%c7] : tensor<8x!ct__D742>
    %extracted_44 = tensor.extract %arg1[%c7] : tensor<8x!ct__D742>
    %ct_45 = cggi.xnor %extracted_43, %extracted_44 : !ct__D742
    %ct_46 = cggi.xnor %ct_42, %ct_45 : !ct__D742
    %ct_47 = cggi.xor %extracted, %extracted_0 : !ct__D742
    %from_elements = tensor.from_elements %ct_47, %ct_6, %ct_13, %ct_20, %ct_27, %ct_34, %ct_41, %ct_46 : tensor<8x!ct__D742>
    return %from_elements : tensor<8x!ct__D742>
  }
}