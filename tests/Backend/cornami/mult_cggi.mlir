#unspecified_bit_field_encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>
module attributes {scheme.cggi} {
  func.func private @internal_generic_1080529143488057702(%arg0: i8, %arg1: i8) -> i8 {
    %0 = arith.muli %arg0, %arg1 : i8
    return %0 : i8
  }
  func.func @test_int_mul(%arg0: memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>, %arg1: memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>) -> memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>> {
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %ct = memref.load %arg0[%c0] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_0 = cggi.not %ct : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_1 = memref.load %arg1[%c0] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_2 = cggi.and %ct, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_3 = memref.load %arg1[%c1] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_4 = cggi.nand %ct, %ct_3 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_5 = memref.load %arg0[%c1] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_6 = cggi.and %ct_5, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_7 = cggi.and %ct_3, %ct_5 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_8 = cggi.and %ct_2, %ct_7 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_9 = cggi.xnor %ct_4, %ct_6 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_10 = memref.load %arg1[%c2] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_11 = cggi.and %ct, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_12 = memref.load %arg0[%c2] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_13 = cggi.and %ct_12, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_14 = cggi.and %ct_3, %ct_12 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_15 = cggi.nand %ct_7, %ct_13 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_16 = cggi.xor %ct_7, %ct_13 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_17 = cggi.nand %ct_11, %ct_16 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_18 = cggi.xor %ct_11, %ct_16 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_19 = cggi.and %ct_8, %ct_18 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_20 = cggi.xor %ct_8, %ct_18 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_21 = memref.load %arg1[%c3] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_22 = cggi.and %ct, %ct_21 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_23 = cggi.nand %ct_15, %ct_17 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_24 = cggi.and %ct_5, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_25 = memref.load %arg0[%c3] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_26 = cggi.and %ct_25, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_27 = cggi.and %ct_3, %ct_25 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_28 = cggi.nand %ct_14, %ct_26 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_29 = cggi.xor %ct_14, %ct_26 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_30 = cggi.nand %ct_24, %ct_29 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_31 = cggi.xor %ct_24, %ct_29 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_32 = cggi.nand %ct_23, %ct_31 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_33 = cggi.xor %ct_23, %ct_31 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_34 = cggi.nand %ct_22, %ct_33 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_35 = cggi.xor %ct_22, %ct_33 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_36 = cggi.and %ct_19, %ct_35 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_37 = cggi.xor %ct_19, %ct_35 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_38 = cggi.nand %ct_32, %ct_34 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_39 = cggi.nand %ct_28, %ct_30 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_40 = cggi.and %ct_12, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_41 = memref.load %arg0[%c4] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_42 = cggi.and %ct_41, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_43 = cggi.and %ct_3, %ct_41 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_44 = cggi.nand %ct_27, %ct_42 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_45 = cggi.xor %ct_27, %ct_42 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_46 = cggi.nand %ct_40, %ct_45 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_47 = cggi.xor %ct_40, %ct_45 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_48 = cggi.nand %ct_39, %ct_47 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_49 = cggi.xor %ct_39, %ct_47 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_50 = cggi.nand %ct_5, %ct_21 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_51 = memref.load %arg1[%c4] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_52 = cggi.and %ct, %ct_51 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_53 = cggi.and %ct_5, %ct_51 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_54 = cggi.and %ct_22, %ct_53 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_55 = cggi.xnor %ct_50, %ct_52 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_56 = cggi.nand %ct_49, %ct_55 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_57 = cggi.xor %ct_49, %ct_55 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_58 = cggi.and %ct_38, %ct_57 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_59 = cggi.xor %ct_38, %ct_57 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_60 = cggi.and %ct_36, %ct_59 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_61 = cggi.xor %ct_36, %ct_59 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_62 = cggi.nand %ct_48, %ct_56 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_63 = memref.load %arg1[%c5] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_64 = cggi.and %ct, %ct_63 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_65 = cggi.and %ct_12, %ct_21 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_66 = cggi.nand %ct_12, %ct_51 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_67 = cggi.nand %ct_53, %ct_65 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_68 = cggi.xor %ct_53, %ct_65 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_69 = cggi.nand %ct_64, %ct_68 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_70 = cggi.xor %ct_64, %ct_68 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_71 = cggi.nand %ct_44, %ct_46 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_72 = cggi.and %ct_25, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_73 = memref.load %arg0[%c5] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_74 = cggi.and %ct_73, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_75 = cggi.nand %ct_3, %ct_73 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_76 = cggi.nand %ct_43, %ct_74 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_77 = cggi.xor %ct_43, %ct_74 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_78 = cggi.nand %ct_72, %ct_77 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_79 = cggi.xor %ct_72, %ct_77 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_80 = cggi.nand %ct_71, %ct_79 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_81 = cggi.xor %ct_71, %ct_79 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_82 = cggi.nand %ct_70, %ct_81 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_83 = cggi.xor %ct_70, %ct_81 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_84 = cggi.nand %ct_62, %ct_83 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_85 = cggi.xor %ct_62, %ct_83 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_86 = cggi.nand %ct_54, %ct_85 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_87 = cggi.xor %ct_54, %ct_85 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_88 = cggi.and %ct_58, %ct_87 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_89 = cggi.xor %ct_58, %ct_87 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_90 = cggi.and %ct_60, %ct_89 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_91 = cggi.nand %ct_84, %ct_86 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_92 = cggi.nand %ct_80, %ct_82 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_93 = cggi.nand %ct_76, %ct_78 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_94 = cggi.and %ct_41, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_95 = memref.load %arg0[%c6] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_96 = cggi.and %ct_95, %ct_1 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_97 = cggi.and %ct_3, %ct_95 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_98 = cggi.nand %ct_74, %ct_97 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_99 = cggi.xnor %ct_75, %ct_96 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_100 = cggi.nand %ct_94, %ct_99 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_101 = cggi.xor %ct_94, %ct_99 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_102 = cggi.nand %ct_93, %ct_101 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_103 = cggi.xor %ct_93, %ct_101 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_104 = cggi.and %ct_5, %ct_63 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_105 = cggi.and %ct_25, %ct_21 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_106 = cggi.and %ct_25, %ct_51 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_107 = cggi.nand %ct_65, %ct_106 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_108 = cggi.xnor %ct_66, %ct_105 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_109 = cggi.nand %ct_104, %ct_108 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_110 = cggi.xor %ct_104, %ct_108 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_111 = cggi.nand %ct_103, %ct_110 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_112 = cggi.xor %ct_103, %ct_110 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_113 = cggi.nand %ct_92, %ct_112 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_114 = cggi.xor %ct_92, %ct_112 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_115 = cggi.nand %ct_67, %ct_69 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_116 = memref.load %arg1[%c6] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_117 = cggi.and %ct, %ct_116 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_118 = cggi.and %ct_115, %ct_117 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_119 = cggi.xor %ct_115, %ct_117 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_120 = cggi.nand %ct_114, %ct_119 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_121 = cggi.xor %ct_114, %ct_119 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_122 = cggi.and %ct_91, %ct_121 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_123 = cggi.xor %ct_91, %ct_121 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_124 = cggi.nand %ct_88, %ct_123 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_125 = cggi.xor %ct_88, %ct_123 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_126 = cggi.nand %ct_90, %ct_125 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_127 = cggi.xor %ct_90, %ct_125 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_128 = cggi.nand %ct_124, %ct_126 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_129 = cggi.and %ct_113, %ct_120 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_130 = cggi.and %ct_98, %ct_100 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_131 = cggi.xnor %ct_106, %ct_130 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_132 = cggi.nand %ct_12, %ct_63 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_133 = cggi.and %ct_5, %ct_116 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_134 = cggi.xnor %ct_132, %ct_133 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_135 = memref.load %arg1[%c7] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_136 = cggi.and %ct_0, %ct_135 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_137 = memref.load %arg0[%c7] : memref<8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %ct_138 = cggi.and %ct_1, %ct_137 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_139 = cggi.xnor %ct_136, %ct_138 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_140 = cggi.xnor %ct_134, %ct_139 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_141 = cggi.and %ct_73, %ct_10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_142 = cggi.and %ct_41, %ct_21 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_143 = cggi.xnor %ct_141, %ct_142 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_144 = cggi.xor %ct_135, %ct_97 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_145 = cggi.xnor %ct_143, %ct_144 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_146 = cggi.xnor %ct_140, %ct_145 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_147 = cggi.xnor %ct_131, %ct_146 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_148 = cggi.nand %ct_107, %ct_109 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_149 = cggi.and %ct_102, %ct_111 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_150 = cggi.xnor %ct_148, %ct_149 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_151 = cggi.xnor %ct_147, %ct_150 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_152 = cggi.xnor %ct_129, %ct_151 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_153 = cggi.xnor %ct_118, %ct_122 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_154 = cggi.xnor %ct_152, %ct_153 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_155 = cggi.xnor %ct_128, %ct_154 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_156 = cggi.xor %ct_60, %ct_89 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
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
