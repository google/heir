#unspecified_bit_field_encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>
module attributes {scheme.cggi} {
  func.func private @internal_generic_3421207664396770948(%arg0: i8, %arg1: i8) -> i8 {
    %0 = arith.muli %arg0, %arg1 : i8
    return %0 : i8
  }
  func.func @test_int_mul(%arg0: !scifrbool.bootstrap_key_standard, %arg1: !scifrbool.key_switch_key, %arg2: !scifrbool.server_parameters, %arg3: memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>, %arg4: memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>) -> memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>> {
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %ct = memref.load %arg3[%c0] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_0 = scifrbool.section(%ct) {
      %ct_157 = scifrbool.not %ct : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_1 = memref.load %arg4[%c0] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_2 = scifrbool.section(%ct, %ct_1) {
      %ct_157 = scifrbool.and %ct, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_3 = memref.load %arg4[%c1] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_4 = scifrbool.section(%ct, %ct_3) {
      %ct_157 = scifrbool.nand %ct, %ct_3 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_5 = memref.load %arg3[%c1] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_6 = scifrbool.section(%ct_5, %ct_1) {
      %ct_157 = scifrbool.and %ct_5, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_7 = scifrbool.section(%ct_3, %ct_5) {
      %ct_157 = scifrbool.and %ct_3, %ct_5 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_8 = scifrbool.section(%ct_2, %ct_7) {
      %ct_157 = scifrbool.and %ct_2, %ct_7 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_9 = scifrbool.section(%ct_4, %ct_6) {
      %ct_157 = scifrbool.xnor %ct_4, %ct_6 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_10 = memref.load %arg4[%c2] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_11 = scifrbool.section(%ct, %ct_10) {
      %ct_157 = scifrbool.and %ct, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_12 = memref.load %arg3[%c2] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_13 = scifrbool.section(%ct_12, %ct_1) {
      %ct_157 = scifrbool.and %ct_12, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_14 = scifrbool.section(%ct_3, %ct_12) {
      %ct_157 = scifrbool.and %ct_3, %ct_12 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_15 = scifrbool.section(%ct_7, %ct_13) {
      %ct_157 = scifrbool.nand %ct_7, %ct_13 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_16 = scifrbool.section(%ct_7, %ct_13) {
      %ct_157 = scifrbool.xor %ct_7, %ct_13 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_17 = scifrbool.section(%ct_11, %ct_16) {
      %ct_157 = scifrbool.nand %ct_11, %ct_16 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_18 = scifrbool.section(%ct_11, %ct_16) {
      %ct_157 = scifrbool.xor %ct_11, %ct_16 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_19 = scifrbool.section(%ct_8, %ct_18) {
      %ct_157 = scifrbool.and %ct_8, %ct_18 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_20 = scifrbool.section(%ct_8, %ct_18) {
      %ct_157 = scifrbool.xor %ct_8, %ct_18 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_21 = memref.load %arg4[%c3] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_22 = scifrbool.section(%ct, %ct_21) {
      %ct_157 = scifrbool.and %ct, %ct_21 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_23 = scifrbool.section(%ct_15, %ct_17) {
      %ct_157 = scifrbool.nand %ct_15, %ct_17 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_24 = scifrbool.section(%ct_5, %ct_10) {
      %ct_157 = scifrbool.and %ct_5, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_25 = memref.load %arg3[%c3] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_26 = scifrbool.section(%ct_25, %ct_1) {
      %ct_157 = scifrbool.and %ct_25, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_27 = scifrbool.section(%ct_3, %ct_25) {
      %ct_157 = scifrbool.and %ct_3, %ct_25 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_28 = scifrbool.section(%ct_14, %ct_26) {
      %ct_157 = scifrbool.nand %ct_14, %ct_26 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_29 = scifrbool.section(%ct_14, %ct_26) {
      %ct_157 = scifrbool.xor %ct_14, %ct_26 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_30 = scifrbool.section(%ct_24, %ct_29) {
      %ct_157 = scifrbool.nand %ct_24, %ct_29 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_31 = scifrbool.section(%ct_24, %ct_29) {
      %ct_157 = scifrbool.xor %ct_24, %ct_29 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_32 = scifrbool.section(%ct_23, %ct_31) {
      %ct_157 = scifrbool.nand %ct_23, %ct_31 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_33 = scifrbool.section(%ct_23, %ct_31) {
      %ct_157 = scifrbool.xor %ct_23, %ct_31 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_34 = scifrbool.section(%ct_22, %ct_33) {
      %ct_157 = scifrbool.nand %ct_22, %ct_33 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_35 = scifrbool.section(%ct_22, %ct_33) {
      %ct_157 = scifrbool.xor %ct_22, %ct_33 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_36 = scifrbool.section(%ct_19, %ct_35) {
      %ct_157 = scifrbool.and %ct_19, %ct_35 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_37 = scifrbool.section(%ct_19, %ct_35) {
      %ct_157 = scifrbool.xor %ct_19, %ct_35 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_38 = scifrbool.section(%ct_32, %ct_34) {
      %ct_157 = scifrbool.nand %ct_32, %ct_34 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_39 = scifrbool.section(%ct_28, %ct_30) {
      %ct_157 = scifrbool.nand %ct_28, %ct_30 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_40 = scifrbool.section(%ct_12, %ct_10) {
      %ct_157 = scifrbool.and %ct_12, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_41 = memref.load %arg3[%c4] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_42 = scifrbool.section(%ct_41, %ct_1) {
      %ct_157 = scifrbool.and %ct_41, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_43 = scifrbool.section(%ct_3, %ct_41) {
      %ct_157 = scifrbool.and %ct_3, %ct_41 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_44 = scifrbool.section(%ct_27, %ct_42) {
      %ct_157 = scifrbool.nand %ct_27, %ct_42 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_45 = scifrbool.section(%ct_27, %ct_42) {
      %ct_157 = scifrbool.xor %ct_27, %ct_42 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_46 = scifrbool.section(%ct_40, %ct_45) {
      %ct_157 = scifrbool.nand %ct_40, %ct_45 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_47 = scifrbool.section(%ct_40, %ct_45) {
      %ct_157 = scifrbool.xor %ct_40, %ct_45 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_48 = scifrbool.section(%ct_39, %ct_47) {
      %ct_157 = scifrbool.nand %ct_39, %ct_47 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_49 = scifrbool.section(%ct_39, %ct_47) {
      %ct_157 = scifrbool.xor %ct_39, %ct_47 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_50 = scifrbool.section(%ct_5, %ct_21) {
      %ct_157 = scifrbool.nand %ct_5, %ct_21 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_51 = memref.load %arg4[%c4] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_52 = scifrbool.section(%ct, %ct_51) {
      %ct_157 = scifrbool.and %ct, %ct_51 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_53 = scifrbool.section(%ct_5, %ct_51) {
      %ct_157 = scifrbool.and %ct_5, %ct_51 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_54 = scifrbool.section(%ct_22, %ct_53) {
      %ct_157 = scifrbool.and %ct_22, %ct_53 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_55 = scifrbool.section(%ct_50, %ct_52) {
      %ct_157 = scifrbool.xnor %ct_50, %ct_52 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_56 = scifrbool.section(%ct_49, %ct_55) {
      %ct_157 = scifrbool.nand %ct_49, %ct_55 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_57 = scifrbool.section(%ct_49, %ct_55) {
      %ct_157 = scifrbool.xor %ct_49, %ct_55 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_58 = scifrbool.section(%ct_38, %ct_57) {
      %ct_157 = scifrbool.and %ct_38, %ct_57 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_59 = scifrbool.section(%ct_38, %ct_57) {
      %ct_157 = scifrbool.xor %ct_38, %ct_57 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_60 = scifrbool.section(%ct_36, %ct_59) {
      %ct_157 = scifrbool.and %ct_36, %ct_59 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_61 = scifrbool.section(%ct_36, %ct_59) {
      %ct_157 = scifrbool.xor %ct_36, %ct_59 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_62 = scifrbool.section(%ct_48, %ct_56) {
      %ct_157 = scifrbool.nand %ct_48, %ct_56 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_63 = memref.load %arg4[%c5] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_64 = scifrbool.section(%ct, %ct_63) {
      %ct_157 = scifrbool.and %ct, %ct_63 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_65 = scifrbool.section(%ct_12, %ct_21) {
      %ct_157 = scifrbool.and %ct_12, %ct_21 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_66 = scifrbool.section(%ct_12, %ct_51) {
      %ct_157 = scifrbool.nand %ct_12, %ct_51 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_67 = scifrbool.section(%ct_53, %ct_65) {
      %ct_157 = scifrbool.nand %ct_53, %ct_65 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_68 = scifrbool.section(%ct_53, %ct_65) {
      %ct_157 = scifrbool.xor %ct_53, %ct_65 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_69 = scifrbool.section(%ct_64, %ct_68) {
      %ct_157 = scifrbool.nand %ct_64, %ct_68 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_70 = scifrbool.section(%ct_64, %ct_68) {
      %ct_157 = scifrbool.xor %ct_64, %ct_68 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_71 = scifrbool.section(%ct_44, %ct_46) {
      %ct_157 = scifrbool.nand %ct_44, %ct_46 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_72 = scifrbool.section(%ct_25, %ct_10) {
      %ct_157 = scifrbool.and %ct_25, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_73 = memref.load %arg3[%c5] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_74 = scifrbool.section(%ct_73, %ct_1) {
      %ct_157 = scifrbool.and %ct_73, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_75 = scifrbool.section(%ct_3, %ct_73) {
      %ct_157 = scifrbool.nand %ct_3, %ct_73 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_76 = scifrbool.section(%ct_43, %ct_74) {
      %ct_157 = scifrbool.nand %ct_43, %ct_74 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_77 = scifrbool.section(%ct_43, %ct_74) {
      %ct_157 = scifrbool.xor %ct_43, %ct_74 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_78 = scifrbool.section(%ct_72, %ct_77) {
      %ct_157 = scifrbool.nand %ct_72, %ct_77 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_79 = scifrbool.section(%ct_72, %ct_77) {
      %ct_157 = scifrbool.xor %ct_72, %ct_77 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_80 = scifrbool.section(%ct_71, %ct_79) {
      %ct_157 = scifrbool.nand %ct_71, %ct_79 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_81 = scifrbool.section(%ct_71, %ct_79) {
      %ct_157 = scifrbool.xor %ct_71, %ct_79 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_82 = scifrbool.section(%ct_70, %ct_81) {
      %ct_157 = scifrbool.nand %ct_70, %ct_81 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_83 = scifrbool.section(%ct_70, %ct_81) {
      %ct_157 = scifrbool.xor %ct_70, %ct_81 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_84 = scifrbool.section(%ct_62, %ct_83) {
      %ct_157 = scifrbool.nand %ct_62, %ct_83 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_85 = scifrbool.section(%ct_62, %ct_83) {
      %ct_157 = scifrbool.xor %ct_62, %ct_83 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_86 = scifrbool.section(%ct_54, %ct_85) {
      %ct_157 = scifrbool.nand %ct_54, %ct_85 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_87 = scifrbool.section(%ct_54, %ct_85) {
      %ct_157 = scifrbool.xor %ct_54, %ct_85 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_88 = scifrbool.section(%ct_58, %ct_87) {
      %ct_157 = scifrbool.and %ct_58, %ct_87 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_89 = scifrbool.section(%ct_58, %ct_87) {
      %ct_157 = scifrbool.xor %ct_58, %ct_87 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_90 = scifrbool.section(%ct_60, %ct_89) {
      %ct_157 = scifrbool.and %ct_60, %ct_89 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_91 = scifrbool.section(%ct_84, %ct_86) {
      %ct_157 = scifrbool.nand %ct_84, %ct_86 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_92 = scifrbool.section(%ct_80, %ct_82) {
      %ct_157 = scifrbool.nand %ct_80, %ct_82 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_93 = scifrbool.section(%ct_76, %ct_78) {
      %ct_157 = scifrbool.nand %ct_76, %ct_78 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_94 = scifrbool.section(%ct_41, %ct_10) {
      %ct_157 = scifrbool.and %ct_41, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_95 = memref.load %arg3[%c6] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_96 = scifrbool.section(%ct_95, %ct_1) {
      %ct_157 = scifrbool.and %ct_95, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_97 = scifrbool.section(%ct_3, %ct_95) {
      %ct_157 = scifrbool.and %ct_3, %ct_95 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_98 = scifrbool.section(%ct_74, %ct_97) {
      %ct_157 = scifrbool.nand %ct_74, %ct_97 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_99 = scifrbool.section(%ct_75, %ct_96) {
      %ct_157 = scifrbool.xnor %ct_75, %ct_96 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_100 = scifrbool.section(%ct_94, %ct_99) {
      %ct_157 = scifrbool.nand %ct_94, %ct_99 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_101 = scifrbool.section(%ct_94, %ct_99) {
      %ct_157 = scifrbool.xor %ct_94, %ct_99 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_102 = scifrbool.section(%ct_93, %ct_101) {
      %ct_157 = scifrbool.nand %ct_93, %ct_101 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_103 = scifrbool.section(%ct_93, %ct_101) {
      %ct_157 = scifrbool.xor %ct_93, %ct_101 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_104 = scifrbool.section(%ct_5, %ct_63) {
      %ct_157 = scifrbool.and %ct_5, %ct_63 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_105 = scifrbool.section(%ct_25, %ct_21) {
      %ct_157 = scifrbool.and %ct_25, %ct_21 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_106 = scifrbool.section(%ct_25, %ct_51) {
      %ct_157 = scifrbool.and %ct_25, %ct_51 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_107 = scifrbool.section(%ct_65, %ct_106) {
      %ct_157 = scifrbool.nand %ct_65, %ct_106 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_108 = scifrbool.section(%ct_66, %ct_105) {
      %ct_157 = scifrbool.xnor %ct_66, %ct_105 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_109 = scifrbool.section(%ct_104, %ct_108) {
      %ct_157 = scifrbool.nand %ct_104, %ct_108 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_110 = scifrbool.section(%ct_104, %ct_108) {
      %ct_157 = scifrbool.xor %ct_104, %ct_108 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_111 = scifrbool.section(%ct_103, %ct_110) {
      %ct_157 = scifrbool.nand %ct_103, %ct_110 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_112 = scifrbool.section(%ct_103, %ct_110) {
      %ct_157 = scifrbool.xor %ct_103, %ct_110 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_113 = scifrbool.section(%ct_92, %ct_112) {
      %ct_157 = scifrbool.nand %ct_92, %ct_112 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_114 = scifrbool.section(%ct_92, %ct_112) {
      %ct_157 = scifrbool.xor %ct_92, %ct_112 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_115 = scifrbool.section(%ct_67, %ct_69) {
      %ct_157 = scifrbool.nand %ct_67, %ct_69 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_116 = memref.load %arg4[%c6] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_117 = scifrbool.section(%ct, %ct_116) {
      %ct_157 = scifrbool.and %ct, %ct_116 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_118 = scifrbool.section(%ct_115, %ct_117) {
      %ct_157 = scifrbool.and %ct_115, %ct_117 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_119 = scifrbool.section(%ct_115, %ct_117) {
      %ct_157 = scifrbool.xor %ct_115, %ct_117 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_120 = scifrbool.section(%ct_114, %ct_119) {
      %ct_157 = scifrbool.nand %ct_114, %ct_119 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_121 = scifrbool.section(%ct_114, %ct_119) {
      %ct_157 = scifrbool.xor %ct_114, %ct_119 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_122 = scifrbool.section(%ct_91, %ct_121) {
      %ct_157 = scifrbool.and %ct_91, %ct_121 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_123 = scifrbool.section(%ct_91, %ct_121) {
      %ct_157 = scifrbool.xor %ct_91, %ct_121 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_124 = scifrbool.section(%ct_88, %ct_123) {
      %ct_157 = scifrbool.nand %ct_88, %ct_123 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_125 = scifrbool.section(%ct_88, %ct_123) {
      %ct_157 = scifrbool.xor %ct_88, %ct_123 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_126 = scifrbool.section(%ct_90, %ct_125) {
      %ct_157 = scifrbool.nand %ct_90, %ct_125 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_127 = scifrbool.section(%ct_90, %ct_125) {
      %ct_157 = scifrbool.xor %ct_90, %ct_125 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_128 = scifrbool.section(%ct_124, %ct_126) {
      %ct_157 = scifrbool.nand %ct_124, %ct_126 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_129 = scifrbool.section(%ct_113, %ct_120) {
      %ct_157 = scifrbool.and %ct_113, %ct_120 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_130 = scifrbool.section(%ct_98, %ct_100) {
      %ct_157 = scifrbool.and %ct_98, %ct_100 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_131 = scifrbool.section(%ct_106, %ct_130) {
      %ct_157 = scifrbool.xnor %ct_106, %ct_130 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_132 = scifrbool.section(%ct_12, %ct_63) {
      %ct_157 = scifrbool.nand %ct_12, %ct_63 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_133 = scifrbool.section(%ct_5, %ct_116) {
      %ct_157 = scifrbool.and %ct_5, %ct_116 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_134 = scifrbool.section(%ct_132, %ct_133) {
      %ct_157 = scifrbool.xnor %ct_132, %ct_133 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_135 = memref.load %arg4[%c7] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_136 = scifrbool.section(%ct_0, %ct_135) {
      %ct_157 = scifrbool.and %ct_0, %ct_135 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_137 = memref.load %arg3[%c7] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_138 = scifrbool.section(%ct_1, %ct_137) {
      %ct_157 = scifrbool.and %ct_1, %ct_137 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_139 = scifrbool.section(%ct_136, %ct_138) {
      %ct_157 = scifrbool.xnor %ct_136, %ct_138 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_140 = scifrbool.section(%ct_134, %ct_139) {
      %ct_157 = scifrbool.xnor %ct_134, %ct_139 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_141 = scifrbool.section(%ct_73, %ct_10) {
      %ct_157 = scifrbool.and %ct_73, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_142 = scifrbool.section(%ct_41, %ct_21) {
      %ct_157 = scifrbool.and %ct_41, %ct_21 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_143 = scifrbool.section(%ct_141, %ct_142) {
      %ct_157 = scifrbool.xnor %ct_141, %ct_142 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_144 = scifrbool.section(%ct_135, %ct_97) {
      %ct_157 = scifrbool.xor %ct_135, %ct_97 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_145 = scifrbool.section(%ct_143, %ct_144) {
      %ct_157 = scifrbool.xnor %ct_143, %ct_144 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_146 = scifrbool.section(%ct_140, %ct_145) {
      %ct_157 = scifrbool.xnor %ct_140, %ct_145 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_147 = scifrbool.section(%ct_131, %ct_146) {
      %ct_157 = scifrbool.xnor %ct_131, %ct_146 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_148 = scifrbool.section(%ct_107, %ct_109) {
      %ct_157 = scifrbool.nand %ct_107, %ct_109 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_149 = scifrbool.section(%ct_102, %ct_111) {
      %ct_157 = scifrbool.and %ct_102, %ct_111 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_150 = scifrbool.section(%ct_148, %ct_149) {
      %ct_157 = scifrbool.xnor %ct_148, %ct_149 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_151 = scifrbool.section(%ct_147, %ct_150) {
      %ct_157 = scifrbool.xnor %ct_147, %ct_150 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_152 = scifrbool.section(%ct_129, %ct_151) {
      %ct_157 = scifrbool.xnor %ct_129, %ct_151 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_153 = scifrbool.section(%ct_118, %ct_122) {
      %ct_157 = scifrbool.xnor %ct_118, %ct_122 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_154 = scifrbool.section(%ct_152, %ct_153) {
      %ct_157 = scifrbool.xnor %ct_152, %ct_153 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_155 = scifrbool.section(%ct_128, %ct_154) {
      %ct_157 = scifrbool.xnor %ct_128, %ct_154 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_156 = scifrbool.section(%ct_60, %ct_89) {
      %ct_157 = scifrbool.xor %ct_60, %ct_89 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %alloc = memref.alloc() : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %ct_2, %alloc[%c0] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %ct_9, %alloc[%c1] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %ct_20, %alloc[%c2] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %ct_37, %alloc[%c3] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %ct_61, %alloc[%c4] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %ct_156, %alloc[%c5] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %ct_127, %alloc[%c6] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %ct_155, %alloc[%c7] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    return %alloc : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
  }
}
