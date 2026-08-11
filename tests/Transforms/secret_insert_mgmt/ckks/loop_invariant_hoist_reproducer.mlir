// RUN: heir-opt --secret-insert-mgmt-ckks="after-mul=true before-mul-include-first-mul=false bootstrap-waterline=0 level-budget=11 min-slot-count=4096" %s | FileCheck %s

// This test (whose assertions are toward the end) ensures that when a loop
// that arises from a convolution starts at level 0, the bootstraps are hoisted
// outside the loop rather than occurring inside the loop for every iteration.

#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 16 = 0 and 0 <= i1 <= 12 and 0 <= slot <= 4095 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x13xf32>, layout = #layout>
module attributes {backend.lattigo, scheme.ckks} {
  func.func private @_assign_layout_16889166383960922983() -> tensor<64x4096xf32> attributes {client.pack_func = {func_name = "tcresnet8small"}} {
    %cst = arith.constant dense_resource<__elided__> : tensor<48x48x6xf32>
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x4096xf32>
    %c64_i32 = arith.constant 64 : i32
    %c6_i32 = arith.constant 6 : i32
    %c319_i32 = arith.constant 319 : i32
    %c63_i32 = arith.constant 63 : i32
    %c512_i32 = arith.constant 512 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %0 = scf.for %arg0 = %c0_i32 to %c48_i32 step %c1_i32 iter_args(%arg1 = %cst_0) -> (tensor<64x4096xf32>)  : i32 {
      %1 = scf.for %arg2 = %c0_i32 to %c48_i32 step %c1_i32 iter_args(%arg3 = %arg1) -> (tensor<64x4096xf32>)  : i32 {
        %2 = scf.for %arg4 = %c0_i32 to %c6_i32 step %c1_i32 iter_args(%arg5 = %arg3) -> (tensor<64x4096xf32>)  : i32 {
          %3 = scf.for %arg6 = %arg0 to %c4096_i32 step %c64_i32 iter_args(%arg7 = %arg5) -> (tensor<64x4096xf32>)  : i32 {
            %4 = arith.muli %arg2, %c6_i32 : i32
            %5 = arith.subi %arg0, %4 : i32
            %6 = arith.subi %5, %arg4 : i32
            %7 = arith.addi %6, %c319_i32 : i32
            %8 = arith.floordivsi %7, %c64_i32 : i32
            %9 = arith.muli %8, %c64_i32 : i32
            %10 = arith.subi %7, %9 : i32
            %11 = arith.addi %10, %4 : i32
            %12 = arith.addi %11, %arg4 : i32
            %13 = arith.subi %12, %arg6 : i32
            %14 = arith.subi %13, %c63_i32 : i32
            %15 = arith.remsi %14, %c512_i32 : i32
            %16 = arith.cmpi eq, %15, %c0_i32 : i32
            %17 = scf.if %16 -> (tensor<64x4096xf32>) {
              %18 = arith.addi %4, %arg4 : i32
              %19 = arith.subi %18, %arg6 : i32
              %20 = arith.addi %19, %c4096_i32 : i32
              %21 = arith.floordivsi %20, %c512_i32 : i32
              %22 = arith.muli %21, %c512_i32 : i32
              %23 = arith.subi %20, %22 : i32
              %24 = arith.index_cast %arg0 : i32 to index
              %25 = arith.index_cast %arg2 : i32 to index
              %26 = arith.index_cast %arg4 : i32 to index
              %extracted = tensor.extract %cst[%24, %25, %26] : tensor<48x48x6xf32>
              %27 = arith.index_cast %23 : i32 to index
              %28 = arith.index_cast %arg6 : i32 to index
              %inserted = tensor.insert %extracted into %arg7[%27, %28] : tensor<64x4096xf32>
              scf.yield %inserted : tensor<64x4096xf32>
            } else {
              scf.yield %arg7 : tensor<64x4096xf32>
            }
            scf.yield %17 : tensor<64x4096xf32>
          }
          scf.yield %3 : tensor<64x4096xf32>
        }
        scf.yield %2 : tensor<64x4096xf32>
      }
      scf.yield %1 : tensor<64x4096xf32>
    }
    return %0 : tensor<64x4096xf32>
  }
  func.func @tcresnet8small(%arg0: !secret.secret<tensor<1x4096xf32>> {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 10, 48>}, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x10x48xf32>, layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (-48i1 - i2 + slot) mod 512 = 0 and 0 <= i1 <= 9 and 0 <= i2 <= 4095 - 48i1 and i2 <= 47 and 0 <= slot <= 4095 and 4096*floor((-512 + 48i1 + i2)/4096) <= -4096 + 48i1 + i2 }">>}) -> (!secret.secret<tensor<1x4096xf32>> {tensor_ext.original_type = #original_type}) {
    %c3904 = arith.constant 3904 : index
    %cst = arith.constant dense_resource<torch_tensor_16_10_3_torch.float32_packed> : tensor<512x4096xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x4096xf32>
    %c0 = arith.constant 0 : index
    %c23 = arith.constant 23 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<512xf64>
    %cst_3 = arith.constant 0.000000e+00 : f64
    %c256 = arith.constant 256 : index
    %c2 = arith.constant 2 : index
    %cst_4 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_24_16_1_torch.float32_packed> : tensor<1024x4096xf32>
    %cst_7 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %cst_8 = arith.constant dense_resource<__elided__> : tensor<1024xf64>
    %cst_9 = arith.constant dense_resource<torch_tensor_24_16_9_torch.float32_packed> : tensor<1024x4096xf32>
    %cst_10 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_11 = arith.constant dense_resource<__elided__> : tensor<1024xf64>
    %cst_12 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_13 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_14 = arith.constant dense_resource<torch_tensor_24_24_9_torch.float32_packed> : tensor<1024x4096xf32>
    %cst_15 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_16 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_17 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_18 = arith.constant dense_resource<torch_tensor_32_24_1_torch.float32_packed> : tensor<512x4096xf32>
    %cst_19 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_20 = arith.constant dense_resource<__elided__> : tensor<512xf64>
    %cst_21 = arith.constant dense_resource<torch_tensor_32_24_9_torch.float32_packed> : tensor<512x4096xf32>
    %cst_22 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_23 = arith.constant dense_resource<__elided__> : tensor<512xf64>
    %cst_24 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_25 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_26 = arith.constant dense_resource<torch_tensor_32_32_9_torch.float32_packed> : tensor<512x4096xf32>
    %cst_27 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_28 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_29 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_30 = arith.constant dense_resource<torch_tensor_48_32_1_torch.float32_packed> : tensor<512x4096xf32>
    %cst_31 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_32 = arith.constant dense_resource<__elided__> : tensor<512xf64>
    %cst_33 = arith.constant dense_resource<torch_tensor_48_32_9_torch.float32_packed> : tensor<512x4096xf32>
    %cst_34 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_35 = arith.constant dense_resource<__elided__> : tensor<512xf64>
    %cst_36 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_37 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_38 = arith.constant dense_resource<torch_tensor_48_48_9_torch.float32_packed> : tensor<512x4096xf32>
    %cst_39 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_40 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_41 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c3 = arith.constant 3 : index
    %cst_42 = arith.constant dense_resource<torch_tensor_13_48_torch.float32_packed> : tensor<16x4096xf32>
    %cst_43 = arith.constant dense_resource<torch_tensor_13_torch.float32_packed> : tensor<1x4096xf32>
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c-23 = arith.constant -23 : index
    %c-32 = arith.constant -32 : index
    %c-8 = arith.constant -8 : index
    %c-4 = arith.constant -4 : index
    %cst_44 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_45 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_46 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_47 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_48 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_49 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_50 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_51 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_52 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_53 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_54 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_55 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_56 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_57 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %c128 = arith.constant 128 : index
    %cst_58 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_59 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_60 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_61 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %c2048 = arith.constant 2048 : index
    %cst_62 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_63 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_64 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_65 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_66 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_67 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_68 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_69 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_70 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_71 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_72 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_73 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_74 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_75 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_76 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_77 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_78 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_79 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_80 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_81 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_82 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_83 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_84 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_85 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_86 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_87 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_88 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_89 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_90 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_91 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_92 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_93 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_94 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_95 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_96 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_97 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_98 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_99 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_100 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_101 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_102 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_103 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_104 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_105 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_106 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_107 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_108 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_109 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_110 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_111 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_112 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_113 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_114 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_115 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_116 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_117 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_118 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_119 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_120 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_121 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_122 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_123 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_124 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_125 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_126 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_127 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_128 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_129 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_130 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_131 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_132 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_133 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_134 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_135 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_136 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_137 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_138 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_139 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_140 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_141 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_142 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_143 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_144 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_145 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_146 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_147 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_148 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_149 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_150 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_151 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_152 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_153 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_154 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_155 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_156 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_157 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_158 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_159 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_160 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_161 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_162 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_163 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_164 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_165 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_166 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_167 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_168 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_169 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_170 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_171 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_172 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_173 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_174 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_175 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_176 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_177 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_178 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_179 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_180 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_181 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_182 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_183 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_184 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_185 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_186 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_187 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_188 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_189 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_190 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_191 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_192 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_193 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_194 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_195 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_196 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_197 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_198 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_199 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_200 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_201 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_202 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_203 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_204 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_205 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_206 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_207 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_208 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_209 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_210 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_211 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_212 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_213 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_214 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_215 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_216 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_217 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_218 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_219 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_220 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_221 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_222 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_223 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_224 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_225 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_226 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_227 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_228 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_229 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_230 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_231 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_232 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_233 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_234 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_235 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_236 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_237 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_238 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_239 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_240 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_241 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_242 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_243 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_244 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_245 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_246 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_247 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_248 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_249 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_250 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_251 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_252 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_253 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_254 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_255 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_256 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_257 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_258 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_259 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_260 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_261 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_262 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_263 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_264 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_265 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_266 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_267 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_268 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %c48 = arith.constant 48 : index
    %cst_269 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_270 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_271 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_272 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_273 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_274 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_275 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_276 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_277 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_278 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_279 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_280 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_281 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_282 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_283 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_284 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_285 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_286 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_287 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_288 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_289 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_290 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_291 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %c3840 = arith.constant 3840 : index
    %c112 = arith.constant 112 : index
    %cst_292 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_293 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %cst_294 = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
    %0 = tensor.empty() : tensor<23x4096xf32>
    %1 = tensor.empty() : tensor<32x4096xf32>
    %2 = call @_assign_layout_16889166383960922983() : () -> tensor<64x4096xf32>
    %3 = tensor.empty() : tensor<8x4096xf32>
    %4 = tensor.empty() : tensor<4x4096xf32>
    %5 = secret.generic(%arg0: !secret.secret<tensor<1x4096xf32>> {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 10, 48>}}) {
    ^body(%input0: tensor<1x4096xf32>):
      debug.validate %input0 {metadata = "input", name = "input"} : tensor<1x4096xf32>
      %6 = arith.mulf %input0, %cst_270 : tensor<1x4096xf32>
      %7 = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2 = %0) -> (tensor<23x4096xf32>) {
        %727 = tensor_ext.rotate %6, %arg1 : tensor<1x4096xf32>, index
        %inserted_slice = tensor.insert_slice %727 into %arg2[%arg1, 0] [1, 4096] [1, 1] : tensor<1x4096xf32> into tensor<23x4096xf32>
        scf.yield %inserted_slice : tensor<23x4096xf32>
      }
      %8 = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2 = %cst_1) -> (tensor<1x4096xf32>) {
        %727 = scf.for %arg3 = %c0 to %c23 step %c1 iter_args(%arg4 = %cst_1) -> (tensor<1x4096xf32>) {
          %731 = arith.muli %arg1, %c23 : index
          %732 = arith.addi %arg3, %731 : index
          %733 = arith.cmpi slt, %732, %c512 : index
          %734 = scf.if %733 -> (tensor<1x4096xf32>) {
            %extracted = tensor.extract %cst_2[%732] : tensor<512xf64>
            %735 = arith.cmpf one, %extracted, %cst_3 : f64
            %736 = scf.if %735 -> (tensor<1x4096xf32>) {
              %extracted_slice = tensor.extract_slice %cst[%732, 0] [1, 4096] [1, 1] : tensor<512x4096xf32> to tensor<1x4096xf32>
              %737 = arith.muli %arg1, %c-23 : index
              %738 = tensor_ext.rotate %extracted_slice, %737 : tensor<1x4096xf32>, index
              %extracted_slice_295 = tensor.extract_slice %7[%arg3, 0] [1, 4096] [1, 1] : tensor<23x4096xf32> to tensor<1x4096xf32>
              %739 = arith.mulf %738, %extracted_slice_295 : tensor<1x4096xf32>
              %740 = arith.addf %arg4, %739 : tensor<1x4096xf32>
              scf.yield %740 : tensor<1x4096xf32>
            } else {
              scf.yield %arg4 : tensor<1x4096xf32>
            }
            scf.yield %736 : tensor<1x4096xf32>
          } else {
            scf.yield %arg4 : tensor<1x4096xf32>
          }
          scf.yield %734 : tensor<1x4096xf32>
        }
        %728 = arith.muli %arg1, %c23 : index
        %729 = tensor_ext.rotate %727, %728 : tensor<1x4096xf32>, index
        %730 = arith.addf %arg2, %729 : tensor<1x4096xf32>
        scf.yield %730 : tensor<1x4096xf32>
      }
      %9 = arith.addf %8, %cst_0 : tensor<1x4096xf32>
      debug.validate %9 {metadata = "conv1", name = "conv1"} : tensor<1x4096xf32>
      %10 = arith.mulf %9, %cst_4 {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 16, 48>}} : tensor<1x4096xf32>
      %11 = arith.addf %10, %cst_5 {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 16, 48>}} : tensor<1x4096xf32>
      %12 = kernel.eval_chebyshev %11 {coefficients = [2.3251965538185049, 3.6698357203950502, 1.7070180567163611, -0.042888578030742995], heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 16, 48>}} : tensor<1x4096xf32> -> tensor<1x4096xf32>
      debug.validate %12 {metadata = "relu1", name = "relu1"} : tensor<1x4096xf32>
      %13 = scf.for %arg1 = %c0 to %c32 step %c1 iter_args(%arg2 = %1) -> (tensor<32x4096xf32>) {
        %727 = tensor_ext.rotate %12, %arg1 : tensor<1x4096xf32>, index
        %inserted_slice = tensor.insert_slice %727 into %arg2[%arg1, 0] [1, 4096] [1, 1] : tensor<1x4096xf32> into tensor<32x4096xf32>
        scf.yield %inserted_slice : tensor<32x4096xf32>
      }
      %14 = scf.for %arg1 = %c0 to %c32 step %c1 iter_args(%arg2 = %cst_1) -> (tensor<1x4096xf32>) {
        %727 = scf.for %arg3 = %c0 to %c32 step %c1 iter_args(%arg4 = %cst_1) -> (tensor<1x4096xf32>) {
          %731 = arith.muli %arg1, %c32 : index
          %732 = arith.addi %arg3, %731 : index
          %733 = arith.cmpi slt, %732, %c1024 : index
          %734 = scf.if %733 -> (tensor<1x4096xf32>) {
            %extracted = tensor.extract %cst_8[%732] : tensor<1024xf64>
            %735 = arith.cmpf one, %extracted, %cst_3 : f64
            %736 = scf.if %735 -> (tensor<1x4096xf32>) {
              %extracted_slice = tensor.extract_slice %cst_6[%732, 0] [1, 4096] [1, 1] : tensor<1024x4096xf32> to tensor<1x4096xf32>
              %737 = arith.muli %arg1, %c-32 : index
              %738 = tensor_ext.rotate %extracted_slice, %737 : tensor<1x4096xf32>, index
              %extracted_slice_295 = tensor.extract_slice %13[%arg3, 0] [1, 4096] [1, 1] : tensor<32x4096xf32> to tensor<1x4096xf32>
              %739 = arith.mulf %738, %extracted_slice_295 : tensor<1x4096xf32>
              %740 = arith.addf %arg4, %739 : tensor<1x4096xf32>
              scf.yield %740 : tensor<1x4096xf32>
            } else {
              scf.yield %arg4 : tensor<1x4096xf32>
            }
            scf.yield %736 : tensor<1x4096xf32>
          } else {
            scf.yield %arg4 : tensor<1x4096xf32>
          }
          scf.yield %734 : tensor<1x4096xf32>
        }
        %728 = arith.muli %arg1, %c32 : index
        %729 = tensor_ext.rotate %727, %728 : tensor<1x4096xf32>, index
        %730 = arith.addf %arg2, %729 : tensor<1x4096xf32>
        scf.yield %730 : tensor<1x4096xf32>
      }
      %15 = arith.addf %14, %cst_7 : tensor<1x4096xf32>
      debug.validate %15 {metadata = "conv2", name = "conv2"} : tensor<1x4096xf32>
      %16 = arith.mulf %12, %cst_271 : tensor<1x4096xf32>
      %17 = scf.for %arg1 = %c0 to %c32 step %c1 iter_args(%arg2 = %1) -> (tensor<32x4096xf32>) {
        %727 = tensor_ext.rotate %16, %arg1 : tensor<1x4096xf32>, index
        %inserted_slice = tensor.insert_slice %727 into %arg2[%arg1, 0] [1, 4096] [1, 1] : tensor<1x4096xf32> into tensor<32x4096xf32>
        scf.yield %inserted_slice : tensor<32x4096xf32>
      }
      %18 = scf.for %arg1 = %c0 to %c32 step %c1 iter_args(%arg2 = %cst_1) -> (tensor<1x4096xf32>) {
        %727 = scf.for %arg3 = %c0 to %c32 step %c1 iter_args(%arg4 = %cst_1) -> (tensor<1x4096xf32>) {
          %731 = arith.muli %arg1, %c32 : index
          %732 = arith.addi %arg3, %731 : index
          %733 = arith.cmpi slt, %732, %c1024 : index
          %734 = scf.if %733 -> (tensor<1x4096xf32>) {
            %extracted = tensor.extract %cst_11[%732] : tensor<1024xf64>
            %735 = arith.cmpf one, %extracted, %cst_3 : f64
            %736 = scf.if %735 -> (tensor<1x4096xf32>) {
              %extracted_slice = tensor.extract_slice %cst_9[%732, 0] [1, 4096] [1, 1] : tensor<1024x4096xf32> to tensor<1x4096xf32>
              %737 = arith.muli %arg1, %c-32 : index
              %738 = tensor_ext.rotate %extracted_slice, %737 : tensor<1x4096xf32>, index
              %extracted_slice_295 = tensor.extract_slice %17[%arg3, 0] [1, 4096] [1, 1] : tensor<32x4096xf32> to tensor<1x4096xf32>
              %739 = arith.mulf %738, %extracted_slice_295 : tensor<1x4096xf32>
              %740 = arith.addf %arg4, %739 : tensor<1x4096xf32>
              scf.yield %740 : tensor<1x4096xf32>
            } else {
              scf.yield %arg4 : tensor<1x4096xf32>
            }
            scf.yield %736 : tensor<1x4096xf32>
          } else {
            scf.yield %arg4 : tensor<1x4096xf32>
          }
          scf.yield %734 : tensor<1x4096xf32>
        }
        %728 = arith.muli %arg1, %c32 : index
        %729 = tensor_ext.rotate %727, %728 : tensor<1x4096xf32>, index
        %730 = arith.addf %arg2, %729 : tensor<1x4096xf32>
        scf.yield %730 : tensor<1x4096xf32>
      }
      %19 = arith.addf %18, %cst_10 : tensor<1x4096xf32>
      debug.validate %19 {metadata = "conv3", name = "conv3"} : tensor<1x4096xf32>
      %20 = arith.mulf %19, %cst_12 {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 24, 24>}} : tensor<1x4096xf32>
      %21 = arith.addf %20, %cst_13 {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 24, 24>}} : tensor<1x4096xf32>
      %22 = kernel.eval_chebyshev %21 {coefficients = [1.928445874218355, 3.1291131356625779, 1.6714159928421066, 0.079184834538447213], heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 24, 24>}} : tensor<1x4096xf32> -> tensor<1x4096xf32>
      debug.validate %22 {metadata = "relu2", name = "relu2"} : tensor<1x4096xf32>
      %23 = arith.mulf %22, %cst_272 : tensor<1x4096xf32>
      %24 = arith.mulf %22, %cst_273 : tensor<1x4096xf32>
      %25 = tensor_ext.rotate %24, %c1 : tensor<1x4096xf32>, index
      %26 = arith.addf %23, %25 : tensor<1x4096xf32>
      %27 = arith.mulf %26, %cst_44 : tensor<1x4096xf32>
      %28 = arith.mulf %26, %cst_45 : tensor<1x4096xf32>
      %29 = tensor_ext.rotate %28, %c2 : tensor<1x4096xf32>, index
      %30 = arith.addf %27, %29 : tensor<1x4096xf32>
      %31 = arith.mulf %30, %cst_46 : tensor<1x4096xf32>
      %32 = arith.mulf %30, %cst_47 : tensor<1x4096xf32>
      %33 = tensor_ext.rotate %32, %c4 : tensor<1x4096xf32>, index
      %34 = arith.addf %31, %33 : tensor<1x4096xf32>
      %35 = arith.mulf %34, %cst_48 : tensor<1x4096xf32>
      %36 = arith.mulf %34, %cst_49 : tensor<1x4096xf32>
      %37 = tensor_ext.rotate %36, %c8 : tensor<1x4096xf32>, index
      %38 = arith.addf %35, %37 : tensor<1x4096xf32>
      %39 = arith.mulf %38, %cst_50 : tensor<1x4096xf32>
      %40 = arith.mulf %38, %cst_51 : tensor<1x4096xf32>
      %41 = tensor_ext.rotate %40, %c16 : tensor<1x4096xf32>, index
      %42 = arith.addf %39, %41 : tensor<1x4096xf32>
      %43 = arith.mulf %42, %cst_52 : tensor<1x4096xf32>
      %44 = arith.mulf %42, %cst_53 : tensor<1x4096xf32>
      %45 = tensor_ext.rotate %44, %c32 : tensor<1x4096xf32>, index
      %46 = arith.addf %43, %45 : tensor<1x4096xf32>
      %47 = arith.mulf %46, %cst_54 : tensor<1x4096xf32>
      %48 = arith.mulf %46, %cst_55 : tensor<1x4096xf32>
      %49 = tensor_ext.rotate %48, %c64 : tensor<1x4096xf32>, index
      %50 = arith.addf %47, %49 : tensor<1x4096xf32>
      %51 = arith.mulf %50, %cst_56 : tensor<1x4096xf32>
      %52 = arith.mulf %50, %cst_57 : tensor<1x4096xf32>
      %53 = tensor_ext.rotate %52, %c128 : tensor<1x4096xf32>, index
      %54 = arith.addf %51, %53 : tensor<1x4096xf32>
      %55 = arith.mulf %54, %cst_58 : tensor<1x4096xf32>
      %56 = arith.mulf %54, %cst_59 : tensor<1x4096xf32>
      %57 = tensor_ext.rotate %56, %c256 : tensor<1x4096xf32>, index
      %58 = arith.addf %55, %57 : tensor<1x4096xf32>
      %59 = arith.mulf %58, %cst_58 : tensor<1x4096xf32>
      %60 = arith.mulf %58, %cst_60 : tensor<1x4096xf32>
      %61 = tensor_ext.rotate %60, %c512 : tensor<1x4096xf32>, index
      %62 = arith.addf %59, %61 : tensor<1x4096xf32>
      %63 = arith.mulf %62, %cst_58 : tensor<1x4096xf32>
      %64 = arith.mulf %62, %cst_61 : tensor<1x4096xf32>
      %65 = tensor_ext.rotate %64, %c1024 : tensor<1x4096xf32>, index
      %66 = arith.addf %63, %65 : tensor<1x4096xf32>
      %67 = arith.mulf %66, %cst_58 : tensor<1x4096xf32>
      %68 = arith.mulf %66, %cst_61 : tensor<1x4096xf32>
      %69 = tensor_ext.rotate %68, %c2048 : tensor<1x4096xf32>, index
      %70 = arith.mulf %22, %cst_274 : tensor<1x4096xf32>
      %71 = arith.mulf %22, %cst_275 : tensor<1x4096xf32>
      %72 = tensor_ext.rotate %71, %c1 : tensor<1x4096xf32>, index
      %73 = arith.addf %70, %72 : tensor<1x4096xf32>
      %74 = arith.mulf %73, %cst_62 : tensor<1x4096xf32>
      %75 = arith.mulf %73, %cst_63 : tensor<1x4096xf32>
      %76 = tensor_ext.rotate %75, %c2 : tensor<1x4096xf32>, index
      %77 = arith.addf %74, %76 : tensor<1x4096xf32>
      %78 = arith.mulf %77, %cst_64 : tensor<1x4096xf32>
      %79 = arith.mulf %77, %cst_65 : tensor<1x4096xf32>
      %80 = tensor_ext.rotate %79, %c4 : tensor<1x4096xf32>, index
      %81 = arith.addf %78, %80 : tensor<1x4096xf32>
      %82 = arith.mulf %81, %cst_66 : tensor<1x4096xf32>
      %83 = arith.mulf %81, %cst_67 : tensor<1x4096xf32>
      %84 = tensor_ext.rotate %83, %c8 : tensor<1x4096xf32>, index
      %85 = arith.addf %82, %84 : tensor<1x4096xf32>
      %86 = arith.mulf %85, %cst_68 : tensor<1x4096xf32>
      %87 = arith.mulf %85, %cst_69 : tensor<1x4096xf32>
      %88 = tensor_ext.rotate %87, %c16 : tensor<1x4096xf32>, index
      %89 = arith.addf %86, %88 : tensor<1x4096xf32>
      %90 = arith.mulf %89, %cst_70 : tensor<1x4096xf32>
      %91 = arith.mulf %89, %cst_71 : tensor<1x4096xf32>
      %92 = tensor_ext.rotate %91, %c32 : tensor<1x4096xf32>, index
      %93 = arith.addf %90, %92 : tensor<1x4096xf32>
      %94 = arith.mulf %93, %cst_72 : tensor<1x4096xf32>
      %95 = arith.mulf %93, %cst_73 : tensor<1x4096xf32>
      %96 = tensor_ext.rotate %95, %c64 : tensor<1x4096xf32>, index
      %97 = arith.addf %94, %96 : tensor<1x4096xf32>
      %98 = arith.mulf %97, %cst_74 : tensor<1x4096xf32>
      %99 = arith.mulf %97, %cst_75 : tensor<1x4096xf32>
      %100 = tensor_ext.rotate %99, %c128 : tensor<1x4096xf32>, index
      %101 = arith.addf %98, %100 : tensor<1x4096xf32>
      %102 = arith.mulf %101, %cst_76 : tensor<1x4096xf32>
      %103 = tensor_ext.rotate %102, %c3840 : tensor<1x4096xf32>, index
      %104 = arith.mulf %103, %cst_292 : tensor<1x4096xf32>
      %105 = arith.mulf %22, %cst_276 : tensor<1x4096xf32>
      %106 = arith.mulf %22, %cst_277 : tensor<1x4096xf32>
      %107 = tensor_ext.rotate %106, %c1 : tensor<1x4096xf32>, index
      %108 = arith.addf %105, %107 : tensor<1x4096xf32>
      %109 = arith.mulf %108, %cst_77 : tensor<1x4096xf32>
      %110 = arith.mulf %108, %cst_78 : tensor<1x4096xf32>
      %111 = tensor_ext.rotate %110, %c2 : tensor<1x4096xf32>, index
      %112 = arith.addf %109, %111 : tensor<1x4096xf32>
      %113 = arith.mulf %112, %cst_79 : tensor<1x4096xf32>
      %114 = arith.mulf %112, %cst_80 : tensor<1x4096xf32>
      %115 = tensor_ext.rotate %114, %c4 : tensor<1x4096xf32>, index
      %116 = arith.addf %113, %115 : tensor<1x4096xf32>
      %117 = arith.mulf %116, %cst_81 : tensor<1x4096xf32>
      %118 = arith.mulf %116, %cst_82 : tensor<1x4096xf32>
      %119 = tensor_ext.rotate %118, %c8 : tensor<1x4096xf32>, index
      %120 = arith.addf %117, %119 : tensor<1x4096xf32>
      %121 = arith.mulf %120, %cst_83 : tensor<1x4096xf32>
      %122 = tensor_ext.rotate %121, %c48 : tensor<1x4096xf32>, index
      %123 = arith.mulf %122, %cst_84 : tensor<1x4096xf32>
      %124 = tensor_ext.rotate %121, %c112 : tensor<1x4096xf32>, index
      %125 = arith.mulf %124, %cst_85 : tensor<1x4096xf32>
      %126 = arith.addf %123, %125 : tensor<1x4096xf32>
      %127 = arith.mulf %126, %cst_85 : tensor<1x4096xf32>
      %128 = arith.mulf %126, %cst_84 : tensor<1x4096xf32>
      %129 = tensor_ext.rotate %128, %c128 : tensor<1x4096xf32>, index
      %130 = arith.addf %127, %129 : tensor<1x4096xf32>
      %131 = arith.mulf %130, %cst_86 : tensor<1x4096xf32>
      %132 = tensor_ext.rotate %131, %c3840 : tensor<1x4096xf32>, index
      %133 = arith.mulf %132, %cst_293 : tensor<1x4096xf32>
      %134 = arith.addf %67, %69 : tensor<1x4096xf32>
      %135 = arith.addf %104, %133 : tensor<1x4096xf32>
      %136 = arith.addf %134, %135 : tensor<1x4096xf32>
      %137 = scf.for %arg1 = %c0 to %c32 step %c1 iter_args(%arg2 = %1) -> (tensor<32x4096xf32>) {
        %727 = tensor_ext.rotate %136, %arg1 : tensor<1x4096xf32>, index
        %inserted_slice = tensor.insert_slice %727 into %arg2[%arg1, 0] [1, 4096] [1, 1] : tensor<1x4096xf32> into tensor<32x4096xf32>
        scf.yield %inserted_slice : tensor<32x4096xf32>
      }
      %138 = scf.for %arg1 = %c0 to %c32 step %c1 iter_args(%arg2 = %cst_1) -> (tensor<1x4096xf32>) {
        %727 = scf.for %arg3 = %c0 to %c32 step %c1 iter_args(%arg4 = %cst_1) -> (tensor<1x4096xf32>) {
          %731 = arith.muli %arg1, %c32 : index
          %732 = arith.addi %arg3, %731 : index
          %733 = arith.cmpi slt, %732, %c1024 : index
          %734 = scf.if %733 -> (tensor<1x4096xf32>) {
            %extracted_slice = tensor.extract_slice %cst_14[%732, 0] [1, 4096] [1, 1] : tensor<1024x4096xf32> to tensor<1x4096xf32>
            %735 = arith.muli %arg1, %c-32 : index
            %736 = tensor_ext.rotate %extracted_slice, %735 : tensor<1x4096xf32>, index
            %extracted_slice_295 = tensor.extract_slice %137[%arg3, 0] [1, 4096] [1, 1] : tensor<32x4096xf32> to tensor<1x4096xf32>
            %737 = arith.mulf %736, %extracted_slice_295 : tensor<1x4096xf32>
            %738 = arith.addf %arg4, %737 : tensor<1x4096xf32>
            scf.yield %738 : tensor<1x4096xf32>
          } else {
            scf.yield %arg4 : tensor<1x4096xf32>
          }
          scf.yield %734 : tensor<1x4096xf32>
        }
        %728 = arith.muli %arg1, %c32 : index
        %729 = tensor_ext.rotate %727, %728 : tensor<1x4096xf32>, index
        %730 = arith.addf %arg2, %729 : tensor<1x4096xf32>
        scf.yield %730 : tensor<1x4096xf32>
      }
      %139 = arith.addf %138, %cst_15 : tensor<1x4096xf32>
      debug.validate %139 {metadata = "conv4", name = "conv4"} : tensor<1x4096xf32>
      %140 = arith.mulf %15, %cst_87 : tensor<1x4096xf32>
      %141 = arith.mulf %15, %cst_88 : tensor<1x4096xf32>
      %142 = tensor_ext.rotate %141, %c1 : tensor<1x4096xf32>, index
      %143 = arith.addf %140, %142 : tensor<1x4096xf32>
      %144 = arith.mulf %143, %cst_89 : tensor<1x4096xf32>
      %145 = arith.mulf %143, %cst_90 : tensor<1x4096xf32>
      %146 = tensor_ext.rotate %145, %c2 : tensor<1x4096xf32>, index
      %147 = arith.addf %144, %146 : tensor<1x4096xf32>
      %148 = arith.mulf %147, %cst_91 : tensor<1x4096xf32>
      %149 = arith.mulf %147, %cst_92 : tensor<1x4096xf32>
      %150 = tensor_ext.rotate %149, %c4 : tensor<1x4096xf32>, index
      %151 = arith.addf %148, %150 : tensor<1x4096xf32>
      %152 = arith.mulf %151, %cst_93 : tensor<1x4096xf32>
      %153 = arith.mulf %151, %cst_94 : tensor<1x4096xf32>
      %154 = tensor_ext.rotate %153, %c8 : tensor<1x4096xf32>, index
      %155 = arith.addf %152, %154 : tensor<1x4096xf32>
      %156 = arith.mulf %155, %cst_95 : tensor<1x4096xf32>
      %157 = arith.mulf %155, %cst_96 : tensor<1x4096xf32>
      %158 = tensor_ext.rotate %157, %c16 : tensor<1x4096xf32>, index
      %159 = arith.addf %156, %158 : tensor<1x4096xf32>
      %160 = arith.mulf %159, %cst_97 : tensor<1x4096xf32>
      %161 = arith.mulf %159, %cst_98 : tensor<1x4096xf32>
      %162 = tensor_ext.rotate %161, %c32 : tensor<1x4096xf32>, index
      %163 = arith.addf %160, %162 : tensor<1x4096xf32>
      %164 = arith.mulf %163, %cst_97 : tensor<1x4096xf32>
      %165 = arith.mulf %163, %cst_99 : tensor<1x4096xf32>
      %166 = tensor_ext.rotate %165, %c64 : tensor<1x4096xf32>, index
      %167 = arith.addf %164, %166 : tensor<1x4096xf32>
      %168 = arith.mulf %167, %cst_97 : tensor<1x4096xf32>
      %169 = arith.mulf %167, %cst_100 : tensor<1x4096xf32>
      %170 = tensor_ext.rotate %169, %c128 : tensor<1x4096xf32>, index
      %171 = arith.addf %168, %170 : tensor<1x4096xf32>
      %172 = arith.mulf %171, %cst_97 : tensor<1x4096xf32>
      %173 = arith.mulf %171, %cst_101 : tensor<1x4096xf32>
      %174 = tensor_ext.rotate %173, %c256 : tensor<1x4096xf32>, index
      %175 = arith.addf %172, %174 : tensor<1x4096xf32>
      %176 = arith.mulf %175, %cst_97 : tensor<1x4096xf32>
      %177 = arith.mulf %175, %cst_102 : tensor<1x4096xf32>
      %178 = tensor_ext.rotate %177, %c512 : tensor<1x4096xf32>, index
      %179 = arith.addf %176, %178 : tensor<1x4096xf32>
      %180 = arith.mulf %179, %cst_97 : tensor<1x4096xf32>
      %181 = arith.mulf %179, %cst_103 : tensor<1x4096xf32>
      %182 = tensor_ext.rotate %181, %c1024 : tensor<1x4096xf32>, index
      %183 = arith.addf %180, %182 : tensor<1x4096xf32>
      %184 = arith.mulf %183, %cst_97 : tensor<1x4096xf32>
      %185 = arith.mulf %183, %cst_103 : tensor<1x4096xf32>
      %186 = tensor_ext.rotate %185, %c2048 : tensor<1x4096xf32>, index
      %187 = arith.mulf %15, %cst_104 : tensor<1x4096xf32>
      %188 = arith.mulf %15, %cst_105 : tensor<1x4096xf32>
      %189 = tensor_ext.rotate %188, %c1 : tensor<1x4096xf32>, index
      %190 = arith.addf %187, %189 : tensor<1x4096xf32>
      %191 = arith.mulf %190, %cst_106 : tensor<1x4096xf32>
      %192 = arith.mulf %190, %cst_107 : tensor<1x4096xf32>
      %193 = tensor_ext.rotate %192, %c2 : tensor<1x4096xf32>, index
      %194 = arith.addf %191, %193 : tensor<1x4096xf32>
      %195 = arith.mulf %194, %cst_108 : tensor<1x4096xf32>
      %196 = arith.mulf %194, %cst_109 : tensor<1x4096xf32>
      %197 = tensor_ext.rotate %196, %c4 : tensor<1x4096xf32>, index
      %198 = arith.addf %195, %197 : tensor<1x4096xf32>
      %199 = arith.mulf %198, %cst_110 : tensor<1x4096xf32>
      %200 = arith.mulf %198, %cst_111 : tensor<1x4096xf32>
      %201 = tensor_ext.rotate %200, %c8 : tensor<1x4096xf32>, index
      %202 = arith.addf %199, %201 : tensor<1x4096xf32>
      %203 = arith.mulf %202, %cst_112 : tensor<1x4096xf32>
      %204 = arith.mulf %202, %cst_113 : tensor<1x4096xf32>
      %205 = tensor_ext.rotate %204, %c16 : tensor<1x4096xf32>, index
      %206 = arith.addf %203, %205 : tensor<1x4096xf32>
      %207 = arith.mulf %206, %cst_114 : tensor<1x4096xf32>
      %208 = arith.mulf %206, %cst_115 : tensor<1x4096xf32>
      %209 = tensor_ext.rotate %208, %c32 : tensor<1x4096xf32>, index
      %210 = arith.addf %207, %209 : tensor<1x4096xf32>
      %211 = arith.mulf %210, %cst_114 : tensor<1x4096xf32>
      %212 = arith.mulf %210, %cst_116 : tensor<1x4096xf32>
      %213 = tensor_ext.rotate %212, %c64 : tensor<1x4096xf32>, index
      %214 = arith.addf %211, %213 : tensor<1x4096xf32>
      %215 = arith.mulf %214, %cst_114 : tensor<1x4096xf32>
      %216 = arith.mulf %214, %cst_117 : tensor<1x4096xf32>
      %217 = tensor_ext.rotate %216, %c128 : tensor<1x4096xf32>, index
      %218 = arith.addf %215, %217 : tensor<1x4096xf32>
      %219 = arith.mulf %218, %cst_114 : tensor<1x4096xf32>
      %220 = arith.mulf %218, %cst_118 : tensor<1x4096xf32>
      %221 = tensor_ext.rotate %220, %c256 : tensor<1x4096xf32>, index
      %222 = arith.addf %219, %221 : tensor<1x4096xf32>
      %223 = arith.mulf %222, %cst_114 : tensor<1x4096xf32>
      %224 = arith.mulf %222, %cst_119 : tensor<1x4096xf32>
      %225 = tensor_ext.rotate %224, %c512 : tensor<1x4096xf32>, index
      %226 = arith.addf %223, %225 : tensor<1x4096xf32>
      %227 = arith.mulf %226, %cst_114 : tensor<1x4096xf32>
      %228 = arith.mulf %226, %cst_120 : tensor<1x4096xf32>
      %229 = tensor_ext.rotate %228, %c1024 : tensor<1x4096xf32>, index
      %230 = arith.addf %227, %229 : tensor<1x4096xf32>
      %231 = arith.mulf %230, %cst_114 : tensor<1x4096xf32>
      %232 = arith.mulf %230, %cst_120 : tensor<1x4096xf32>
      %233 = tensor_ext.rotate %232, %c2048 : tensor<1x4096xf32>, index
      %234 = arith.addf %139, %184 : tensor<1x4096xf32>
      %235 = arith.addf %186, %231 : tensor<1x4096xf32>
      %236 = arith.addf %235, %233 : tensor<1x4096xf32>
      %237 = arith.addf %234, %236 : tensor<1x4096xf32>
      %238 = arith.mulf %237, %cst_16 {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 24, 24>}} : tensor<1x4096xf32>
      %239 = arith.addf %238, %cst_17 {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 24, 24>}} : tensor<1x4096xf32>
      %240 = kernel.eval_chebyshev %239 {coefficients = [3.2315526501609444, 5.0763567915464565, 2.3095994252524283, -0.08145488075797401], heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 24, 24>}} : tensor<1x4096xf32> -> tensor<1x4096xf32>
      debug.validate %240 {metadata = "relu3", name = "relu3"} : tensor<1x4096xf32>
      %241 = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2 = %0) -> (tensor<23x4096xf32>) {
        %727 = tensor_ext.rotate %240, %arg1 : tensor<1x4096xf32>, index
        %inserted_slice = tensor.insert_slice %727 into %arg2[%arg1, 0] [1, 4096] [1, 1] : tensor<1x4096xf32> into tensor<23x4096xf32>
        scf.yield %inserted_slice : tensor<23x4096xf32>
      }
      %242 = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2 = %cst_1) -> (tensor<1x4096xf32>) {
        %727 = scf.for %arg3 = %c0 to %c23 step %c1 iter_args(%arg4 = %cst_1) -> (tensor<1x4096xf32>) {
          %731 = arith.muli %arg1, %c23 : index
          %732 = arith.addi %arg3, %731 : index
          %733 = arith.cmpi slt, %732, %c512 : index
          %734 = scf.if %733 -> (tensor<1x4096xf32>) {
            %extracted = tensor.extract %cst_20[%732] : tensor<512xf64>
            %735 = arith.cmpf one, %extracted, %cst_3 : f64
            %736 = scf.if %735 -> (tensor<1x4096xf32>) {
              %extracted_slice = tensor.extract_slice %cst_18[%732, 0] [1, 4096] [1, 1] : tensor<512x4096xf32> to tensor<1x4096xf32>
              %737 = arith.muli %arg1, %c-23 : index
              %738 = tensor_ext.rotate %extracted_slice, %737 : tensor<1x4096xf32>, index
              %extracted_slice_295 = tensor.extract_slice %241[%arg3, 0] [1, 4096] [1, 1] : tensor<23x4096xf32> to tensor<1x4096xf32>
              %739 = arith.mulf %738, %extracted_slice_295 : tensor<1x4096xf32>
              %740 = arith.addf %arg4, %739 : tensor<1x4096xf32>
              scf.yield %740 : tensor<1x4096xf32>
            } else {
              scf.yield %arg4 : tensor<1x4096xf32>
            }
            scf.yield %736 : tensor<1x4096xf32>
          } else {
            scf.yield %arg4 : tensor<1x4096xf32>
          }
          scf.yield %734 : tensor<1x4096xf32>
        }
        %728 = arith.muli %arg1, %c23 : index
        %729 = tensor_ext.rotate %727, %728 : tensor<1x4096xf32>, index
        %730 = arith.addf %arg2, %729 : tensor<1x4096xf32>
        scf.yield %730 : tensor<1x4096xf32>
      }
      %243 = tensor_ext.rotate %242, %c512 : tensor<1x4096xf32>, index
      %244 = arith.addf %242, %cst_19 : tensor<1x4096xf32>
      %245 = arith.addf %244, %243 : tensor<1x4096xf32>
      // This serves as a starting marker for the filecheck matches
      // CHECK: conv5
      debug.validate %245 {metadata = "conv5", name = "conv5"} : tensor<1x4096xf32>
      %246 = arith.mulf %240, %cst_278 : tensor<1x4096xf32>
      %247 = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2 = %0) -> (tensor<23x4096xf32>) {
        %727 = tensor_ext.rotate %246, %arg1 : tensor<1x4096xf32>, index
        %inserted_slice = tensor.insert_slice %727 into %arg2[%arg1, 0] [1, 4096] [1, 1] : tensor<1x4096xf32> into tensor<23x4096xf32>
        scf.yield %inserted_slice : tensor<23x4096xf32>
      }
      // At this point in the program the levels have been reduced to zero, so
      // that a buggy bootstrapping algorithm would insert bootstraps mid-loop,
      // and this would cause the pipeline to think that the loop consumes
      // levels, triggering Halo loops passes and resulting in a program that
      // bootstraps on every loop iteration. The checks here assert that the
      // bootstrap is only inserted before the loop begins, and not during
      // the loop at all.

      // CHECK: %[[BOOT_INPUT:.*]] = mgmt.bootstrap %{{.*}} {mgmt.mgmt = #mgmt.mgmt<level = 11>} : tensor<1x4096xf32>
      // CHECK: %[[LOOP_OUT:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<23x4096xf32>) {
      // CHECK:   %[[ROT:.*]] = tensor_ext.rotate %[[BOOT_INPUT]]
      // CHECK:   %[[INS:.*]] = tensor.insert_slice %[[ROT]]
      // CHECK:   scf.yield %[[INS]]
      // CHECK: }
      // CHECK: %[[PEELED:.*]] = scf.if %{{.*}} -> (tensor<1x4096xf32>) {
      // CHECK:   %[[VAL_10:.*]] = arith.addf
      // CHECK:   scf.yield %[[VAL_10]]
      // CHECK: } else {
      // CHECK:   scf.yield %{{.*}}
      // CHECK: } {mgmt.mgmt = #mgmt.mgmt<level = 10>}
      // CHECK-NOT: mgmt.level_reduce_min
      // CHECK: scf.for %[[ARG1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACCUM:.*]] = %[[PEELED]]) -> (tensor<1x4096xf32>) {
      // CHECK-NOT: mgmt.bootstrap %[[ACCUM]]
      // CHECK:   %[[IF1:.*]] = scf.if %{{.*}} -> (tensor<1x4096xf32>) {
      // CHECK:     %[[IF2:.*]] = scf.if %{{.*}} -> (tensor<1x4096xf32>) {
      // CHECK:       %[[SLICE:.*]] = tensor.extract_slice %[[LOOP_OUT]]
      // CHECK:       %[[MUL:.*]] = arith.mulf %{{.*}}, %[[SLICE]]
      // CHECK:       %[[RESCALE:.*]] = mgmt.modreduce %[[MUL]] {mgmt.mgmt = #mgmt.mgmt<level = 10>}
      // CHECK:       %[[ADD:.*]] = arith.addf %[[ACCUM]], %[[RESCALE]] {mgmt.mgmt = #mgmt.mgmt<level = 10>}
      // CHECK:       scf.yield %[[ADD]]
      // CHECK:     } else {
      // CHECK:       scf.yield %[[ACCUM]]
      // CHECK:     }
      // CHECK:     scf.yield %[[IF2]]
      // CHECK:   } else {
      // CHECK:     scf.yield %[[ACCUM]]
      // CHECK:   }
      // CHECK-NOT: mgmt.level_reduce_min
      // CHECK:   scf.yield %[[IF1]]
      // CHECK: }
      %248 = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2 = %cst_1) -> (tensor<1x4096xf32>) {
        %727 = scf.for %arg3 = %c0 to %c23 step %c1 iter_args(%arg4 = %cst_1) -> (tensor<1x4096xf32>) {
          %731 = arith.muli %arg1, %c23 : index
          %732 = arith.addi %arg3, %731 : index
          %733 = arith.cmpi slt, %732, %c512 : index
          %734 = scf.if %733 -> (tensor<1x4096xf32>) {
            %extracted = tensor.extract %cst_23[%732] : tensor<512xf64>
            %735 = arith.cmpf one, %extracted, %cst_3 : f64
            %736 = scf.if %735 -> (tensor<1x4096xf32>) {
              %extracted_slice = tensor.extract_slice %cst_21[%732, 0] [1, 4096] [1, 1] : tensor<512x4096xf32> to tensor<1x4096xf32>
              %737 = arith.muli %arg1, %c-23 : index
              %738 = tensor_ext.rotate %extracted_slice, %737 : tensor<1x4096xf32>, index
              %extracted_slice_295 = tensor.extract_slice %247[%arg3, 0] [1, 4096] [1, 1] : tensor<23x4096xf32> to tensor<1x4096xf32>
              %739 = arith.mulf %738, %extracted_slice_295 : tensor<1x4096xf32>
              %740 = arith.addf %arg4, %739 : tensor<1x4096xf32>
              scf.yield %740 : tensor<1x4096xf32>
            } else {
              scf.yield %arg4 : tensor<1x4096xf32>
            }
            scf.yield %736 : tensor<1x4096xf32>
          } else {
            scf.yield %arg4 : tensor<1x4096xf32>
          }
          scf.yield %734 : tensor<1x4096xf32>
        }
        %728 = arith.muli %arg1, %c23 : index
        %729 = tensor_ext.rotate %727, %728 : tensor<1x4096xf32>, index
        %730 = arith.addf %arg2, %729 : tensor<1x4096xf32>
        scf.yield %730 : tensor<1x4096xf32>
      }
      %249 = tensor_ext.rotate %248, %c512 : tensor<1x4096xf32>, index
      %250 = arith.addf %248, %cst_22 : tensor<1x4096xf32>
      %251 = arith.addf %250, %249 : tensor<1x4096xf32>
      debug.validate %251 {metadata = "conv6", name = "conv6"} : tensor<1x4096xf32>
      secret.yield %251 : tensor<1x4096xf32>
    } -> !secret.secret<tensor<1x4096xf32>>
    return %5 : !secret.secret<tensor<1x4096xf32>>
  }
}
