// RUN: heir-opt --rotation-analysis --split-input-file %s | FileCheck %s

// Generated with
// heir-opt "--annotate-module=backend=lattigo scheme=ckks" "--mlir-to-ckks=ciphertext-degree=1024 level-budget=2 modulus-switch-after-mul=true experimental-disable-loop-unroll=true first-mod-bits=55" --scheme-to-lattigo --dump-pass-pipeline --mlir-print-ir-before-all --mlir-print-ir-tree-dir=/tmp/mlir $PWD/tests/Examples/common/matvec_512x784.mlir
//
// Then copied as
// cp /tmp/mlir/builtin_module_no-symbol-name/89_lattigo-configure-crypto-context.mlir tests/Transforms/rotation_analysis/large_example.mlir

// CHECK: module attributes
// CHECK-SAME: rotation_analysis.indices = array<i64: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 46, 69, 92, 115, 138, 161, 184, 207, 230, 253, 276, 299, 322, 345, 368, 391, 414, 437, 460, 483, 506, 512>

!bootstrapping_evaluator = !lattigo.ckks.bootstrapping_evaluator
!ct = !lattigo.rlwe.ciphertext
!decryptor = !lattigo.rlwe.decryptor
!encoder = !lattigo.ckks.encoder
!encryptor_pk = !lattigo.rlwe.encryptor<publicKey = true>
!evaluator = !lattigo.ckks.evaluator
!param = !lattigo.ckks.parameter
!pt = !lattigo.rlwe.plaintext
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 511 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<512xf32>, layout = #layout>
module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184372121601], P = [1152921504607338497], logDefaultScale = 45, encryptionTechnique = extended>, scheme.actual_slot_count = 8192 : i64, scheme.ckks, scheme.requested_slot_count = 8192 : i64} {
  func.func private @_assign_layout_4710750956904016321() -> tensor<512x1024xf32> attributes {client.pack_func = {func_name = "matvec"}} {
    %c512_i32 = arith.constant 512 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c240_i32 = arith.constant 240 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<512x1024xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = scf.for %arg0 = %c0_i32 to %c512_i32 step %c1_i32 iter_args(%arg1 = %cst) -> (tensor<512x1024xf32>)  : i32 {
      %1 = scf.for %arg2 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg3 = %arg1) -> (tensor<512x1024xf32>)  : i32 {
        %2 = arith.addi %arg0, %arg2 : i32
        %3 = arith.addi %2, %c240_i32 : i32
        %4 = arith.remsi %3, %c1024_i32 : i32
        %5 = arith.cmpi sge, %4, %c240_i32 : i32
        %6 = scf.if %5 -> (tensor<512x1024xf32>) {
          %7 = arith.index_cast %arg0 : i32 to index
          %8 = arith.index_cast %arg2 : i32 to index
          %inserted = tensor.insert %cst_0 into %arg3[%7, %8] : tensor<512x1024xf32>
          scf.yield %inserted : tensor<512x1024xf32>
        } else {
          scf.yield %arg3 : tensor<512x1024xf32>
        }
        scf.yield %6 : tensor<512x1024xf32>
      }
      scf.yield %1 : tensor<512x1024xf32>
    }
    return %0 : tensor<512x1024xf32>
  }
  func.func @matvec(%bootstrapping_evaluator: !bootstrapping_evaluator, %evaluator: !evaluator, %param: !param, %encoder: !encoder, %arg0: tensor<1x!ct> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<784xf32>, layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 783 and 0 <= slot <= 1023 }">>}, %ct: !ct {client.enc_zero_arg}) -> (tensor<1x!ct> {tensor_ext.original_type = #original_type}) {
    %c-23 = arith.constant -23 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c23 = arith.constant 23 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<1024xf32>
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = call @_assign_layout_4710750956904016321() : () -> tensor<512x1024xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1024xf32>
    %pt = lattigo.ckks.new_plaintext %param : (!param) -> !pt
    %pt_1 = lattigo.ckks.encode %encoder, %extracted_slice, %pt {scale = 45 : i64} : (!encoder, tensor<1024xf32>, !pt) -> !pt
    %extracted = tensor.extract %arg0[%c0] : tensor<1x!ct>
    %ct_2 = lattigo.ckks.mul_new %evaluator, %extracted, %pt_1 : (!evaluator, !ct, !pt) -> !ct
    %ct_3 = lattigo.ckks.rescale_new %evaluator, %ct_2 : (!evaluator, !ct) -> !ct
    %pt_4 = lattigo.ckks.new_plaintext %param : (!param) -> !pt
    %pt_5 = lattigo.ckks.encode %encoder, %cst, %pt_4 {scale = 45 : i64} : (!encoder, tensor<1024xf32>, !pt) -> !pt
    %1 = tensor.empty() : tensor<1x!ct>
    %ct_6 = lattigo.ckks.add_new %evaluator, %ct_3, %pt_5 : (!evaluator, !ct, !pt) -> !ct
    %inserted = tensor.insert %ct_6 into %1[%c0] : tensor<1x!ct>
    %2 = scf.for %arg1 = %c1 to %c23 step %c2 iter_args(%arg2 = %inserted) -> (tensor<1x!ct>) {
      %extracted_14 = tensor.extract %arg2[%c0] : tensor<1x!ct>
      %ct_15 = lattigo.ckks.bootstrap %bootstrapping_evaluator, %extracted_14 : (!bootstrapping_evaluator, !ct) -> !ct
      %4 = arith.cmpi slt, %arg1, %c512 : index
      %5 = scf.if %4 -> (tensor<1x!ct>) {
        %ct_16 = lattigo.ckks.rotate_new %evaluator, %extracted, %arg1 : (!evaluator, !ct, index) -> !ct
        %extracted_slice_17 = tensor.extract_slice %0[%arg1, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1024xf32>
        %pt_18 = lattigo.ckks.new_plaintext %param : (!param) -> !pt
        %pt_19 = lattigo.ckks.encode %encoder, %extracted_slice_17, %pt_18 {scale = 45 : i64} : (!encoder, tensor<1024xf32>, !pt) -> !pt
        %ct_20 = lattigo.ckks.mul_new %evaluator, %ct_16, %pt_19 : (!evaluator, !ct, !pt) -> !ct
        %ct_21 = lattigo.ckks.rescale_new %evaluator, %ct_20 : (!evaluator, !ct) -> !ct
        %pt_22 = lattigo.ckks.new_plaintext %param : (!param) -> !pt
        %pt_23 = lattigo.ckks.encode %encoder, %cst_0, %pt_22 {scale = 45 : i64} : (!encoder, tensor<1024xf32>, !pt) -> !pt
        %ct_24 = lattigo.ckks.mul_new %evaluator, %ct_15, %pt_23 : (!evaluator, !ct, !pt) -> !ct
        %ct_25 = lattigo.ckks.rescale_new %evaluator, %ct_24 : (!evaluator, !ct) -> !ct
        %ct_26 = lattigo.ckks.add_new %evaluator, %ct_25, %ct_21 : (!evaluator, !ct, !ct) -> !ct
        %inserted_27 = tensor.insert %ct_26 into %1[%c0] : tensor<1x!ct>
        scf.yield %inserted_27 : tensor<1x!ct>
      } else {
        %ct_16 = lattigo.rlwe.drop_level_new %evaluator, %ct_15 : (!evaluator, !ct) -> !ct
        %inserted_17 = tensor.insert %ct_16 into %1[%c0] : tensor<1x!ct>
        scf.yield %inserted_17 : tensor<1x!ct>
      }
      %6 = arith.addi %arg1, %c1 : index
      %7 = arith.cmpi slt, %6, %c512 : index
      %8 = scf.if %7 -> (tensor<1x!ct>) {
        %ct_16 = lattigo.ckks.rotate_new %evaluator, %extracted, %6 : (!evaluator, !ct, index) -> !ct
        %extracted_slice_17 = tensor.extract_slice %0[%6, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1024xf32>
        %pt_18 = lattigo.ckks.new_plaintext %param : (!param) -> !pt
        %pt_19 = lattigo.ckks.encode %encoder, %extracted_slice_17, %pt_18 {scale = 45 : i64} : (!encoder, tensor<1024xf32>, !pt) -> !pt
        %ct_20 = lattigo.ckks.mul_new %evaluator, %ct_16, %pt_19 : (!evaluator, !ct, !pt) -> !ct
        %ct_21 = lattigo.ckks.rescale_new %evaluator, %ct_20 : (!evaluator, !ct) -> !ct
        %extracted_22 = tensor.extract %5[%c0] : tensor<1x!ct>
        %ct_23 = lattigo.ckks.add_new %evaluator, %extracted_22, %ct_21 : (!evaluator, !ct, !ct) -> !ct
        %inserted_24 = tensor.insert %ct_23 into %1[%c0] : tensor<1x!ct>
        scf.yield %inserted_24 : tensor<1x!ct>
      } else {
        scf.yield %5 : tensor<1x!ct>
      }
      scf.yield %8 : tensor<1x!ct>
    }
    %extracted_7 = tensor.extract %2[%c0] : tensor<1x!ct>
    %ct_8 = lattigo.ckks.add_new %evaluator, %extracted_7, %pt_5 : (!evaluator, !ct, !pt) -> !ct
    %inserted_9 = tensor.insert %ct_8 into %1[%c0] : tensor<1x!ct>
    %3 = scf.for %arg1 = %c1 to %c23 step %c1 iter_args(%arg2 = %inserted_9) -> (tensor<1x!ct>) {
      %extracted_14 = tensor.extract %arg2[%c0] : tensor<1x!ct>
      %ct_15 = lattigo.ckks.bootstrap %bootstrapping_evaluator, %extracted_14 : (!bootstrapping_evaluator, !ct) -> !ct
      %4 = arith.muli %arg1, %c23 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      %6 = scf.if %5 -> (tensor<1x!ct>) {
        %8 = arith.muli %arg1, %c-23 : index
        %9 = arith.remsi %8, %c1024 : index
        %10 = arith.addi %9, %c1024 : index
        %11 = arith.remsi %10, %c1024 : index
        %12 = arith.subi %c1024, %11 : index
        %extracted_slice_21 = tensor.extract_slice %0[%4, 0] [1, %11] [1, 1] : tensor<512x1024xf32> to tensor<1x?xf32>
        %extracted_slice_22 = tensor.extract_slice %0[%4, %11] [1, %12] [1, 1] : tensor<512x1024xf32> to tensor<1x?xf32>
        %13 = tensor.empty() : tensor<1x1024xf32>
        %inserted_slice = tensor.insert_slice %extracted_slice_21 into %13[0, %12] [1, %11] [1, 1] : tensor<1x?xf32> into tensor<1x1024xf32>
        %inserted_slice_23 = tensor.insert_slice %extracted_slice_22 into %inserted_slice[0, 0] [1, %12] [1, 1] : tensor<1x?xf32> into tensor<1x1024xf32>
        %extracted_slice_24 = tensor.extract_slice %inserted_slice_23[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
        %pt_25 = lattigo.ckks.new_plaintext %param : (!param) -> !pt
        %pt_26 = lattigo.ckks.encode %encoder, %extracted_slice_24, %pt_25 {scale = 45 : i64} : (!encoder, tensor<1024xf32>, !pt) -> !pt
        %ct_27 = lattigo.ckks.mul_new %evaluator, %extracted, %pt_26 : (!evaluator, !ct, !pt) -> !ct
        %ct_28 = lattigo.ckks.rescale_new %evaluator, %ct_27 : (!evaluator, !ct) -> !ct
        %ct_29 = lattigo.ckks.add_new %evaluator, %ct_28, %pt_5 : (!evaluator, !ct, !pt) -> !ct
        %inserted_30 = tensor.insert %ct_29 into %1[%c0] : tensor<1x!ct>
        scf.yield %inserted_30 : tensor<1x!ct>
      } else {
        %splat = tensor.splat %ct : tensor<1x!ct>
        %extracted_21 = tensor.extract %splat[%c0] : tensor<1x!ct>
        %ct_22 = lattigo.ckks.add_new %evaluator, %extracted_21, %pt_5 : (!evaluator, !ct, !pt) -> !ct
        %inserted_23 = tensor.insert %ct_22 into %1[%c0] : tensor<1x!ct>
        scf.yield %inserted_23 : tensor<1x!ct>
      }
      %7 = scf.for %arg3 = %c1 to %c23 step %c2 iter_args(%arg4 = %6) -> (tensor<1x!ct>) {
        %extracted_21 = tensor.extract %arg4[%c0] : tensor<1x!ct>
        %ct_22 = lattigo.ckks.bootstrap %bootstrapping_evaluator, %extracted_21 : (!bootstrapping_evaluator, !ct) -> !ct
        %8 = arith.addi %arg3, %4 : index
        %9 = arith.cmpi slt, %8, %c512 : index
        %10 = scf.if %9 -> (tensor<1x!ct>) {
          %15 = arith.muli %arg1, %c-23 : index
          %16 = arith.remsi %15, %c1024 : index
          %17 = arith.addi %16, %c1024 : index
          %18 = arith.remsi %17, %c1024 : index
          %19 = arith.subi %c1024, %18 : index
          %extracted_slice_23 = tensor.extract_slice %0[%8, 0] [1, %18] [1, 1] : tensor<512x1024xf32> to tensor<1x?xf32>
          %extracted_slice_24 = tensor.extract_slice %0[%8, %18] [1, %19] [1, 1] : tensor<512x1024xf32> to tensor<1x?xf32>
          %20 = tensor.empty() : tensor<1x1024xf32>
          %inserted_slice = tensor.insert_slice %extracted_slice_23 into %20[0, %19] [1, %18] [1, 1] : tensor<1x?xf32> into tensor<1x1024xf32>
          %inserted_slice_25 = tensor.insert_slice %extracted_slice_24 into %inserted_slice[0, 0] [1, %19] [1, 1] : tensor<1x?xf32> into tensor<1x1024xf32>
          %ct_26 = lattigo.ckks.rotate_new %evaluator, %extracted, %arg3 : (!evaluator, !ct, index) -> !ct
          %extracted_slice_27 = tensor.extract_slice %inserted_slice_25[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
          %pt_28 = lattigo.ckks.new_plaintext %param : (!param) -> !pt
          %pt_29 = lattigo.ckks.encode %encoder, %extracted_slice_27, %pt_28 {scale = 45 : i64} : (!encoder, tensor<1024xf32>, !pt) -> !pt
          %ct_30 = lattigo.ckks.mul_new %evaluator, %ct_26, %pt_29 : (!evaluator, !ct, !pt) -> !ct
          %ct_31 = lattigo.ckks.rescale_new %evaluator, %ct_30 : (!evaluator, !ct) -> !ct
          %pt_32 = lattigo.ckks.new_plaintext %param : (!param) -> !pt
          %pt_33 = lattigo.ckks.encode %encoder, %cst_0, %pt_32 {scale = 45 : i64} : (!encoder, tensor<1024xf32>, !pt) -> !pt
          %ct_34 = lattigo.ckks.mul_new %evaluator, %ct_22, %pt_33 : (!evaluator, !ct, !pt) -> !ct
          %ct_35 = lattigo.ckks.rescale_new %evaluator, %ct_34 : (!evaluator, !ct) -> !ct
          %ct_36 = lattigo.ckks.add_new %evaluator, %ct_35, %ct_31 : (!evaluator, !ct, !ct) -> !ct
          %inserted_37 = tensor.insert %ct_36 into %1[%c0] : tensor<1x!ct>
          scf.yield %inserted_37 : tensor<1x!ct>
        } else {
          %ct_23 = lattigo.rlwe.drop_level_new %evaluator, %ct_22 : (!evaluator, !ct) -> !ct
          %inserted_24 = tensor.insert %ct_23 into %1[%c0] : tensor<1x!ct>
          scf.yield %inserted_24 : tensor<1x!ct>
        }
        %11 = arith.addi %arg3, %c1 : index
        %12 = arith.addi %11, %4 : index
        %13 = arith.cmpi slt, %12, %c512 : index
        %14 = scf.if %13 -> (tensor<1x!ct>) {
          %15 = arith.muli %arg1, %c-23 : index
          %16 = arith.remsi %15, %c1024 : index
          %17 = arith.addi %16, %c1024 : index
          %18 = arith.remsi %17, %c1024 : index
          %19 = arith.subi %c1024, %18 : index
          %extracted_slice_23 = tensor.extract_slice %0[%12, 0] [1, %18] [1, 1] : tensor<512x1024xf32> to tensor<1x?xf32>
          %extracted_slice_24 = tensor.extract_slice %0[%12, %18] [1, %19] [1, 1] : tensor<512x1024xf32> to tensor<1x?xf32>
          %20 = tensor.empty() : tensor<1x1024xf32>
          %inserted_slice = tensor.insert_slice %extracted_slice_23 into %20[0, %19] [1, %18] [1, 1] : tensor<1x?xf32> into tensor<1x1024xf32>
          %inserted_slice_25 = tensor.insert_slice %extracted_slice_24 into %inserted_slice[0, 0] [1, %19] [1, 1] : tensor<1x?xf32> into tensor<1x1024xf32>
          %ct_26 = lattigo.ckks.rotate_new %evaluator, %extracted, %11 : (!evaluator, !ct, index) -> !ct
          %extracted_slice_27 = tensor.extract_slice %inserted_slice_25[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
          %pt_28 = lattigo.ckks.new_plaintext %param : (!param) -> !pt
          %pt_29 = lattigo.ckks.encode %encoder, %extracted_slice_27, %pt_28 {scale = 45 : i64} : (!encoder, tensor<1024xf32>, !pt) -> !pt
          %ct_30 = lattigo.ckks.mul_new %evaluator, %ct_26, %pt_29 : (!evaluator, !ct, !pt) -> !ct
          %ct_31 = lattigo.ckks.rescale_new %evaluator, %ct_30 : (!evaluator, !ct) -> !ct
          %extracted_32 = tensor.extract %10[%c0] : tensor<1x!ct>
          %ct_33 = lattigo.ckks.add_new %evaluator, %extracted_32, %ct_31 : (!evaluator, !ct, !ct) -> !ct
          %inserted_34 = tensor.insert %ct_33 into %1[%c0] : tensor<1x!ct>
          scf.yield %inserted_34 : tensor<1x!ct>
        } else {
          scf.yield %10 : tensor<1x!ct>
        }
        scf.yield %14 : tensor<1x!ct>
      }
      %extracted_16 = tensor.extract %7[%c0] : tensor<1x!ct>
      %ct_17 = lattigo.ckks.rotate_new %evaluator, %extracted_16, %4 : (!evaluator, !ct, index) -> !ct
      %ct_18 = lattigo.rlwe.drop_level_new %evaluator, %ct_15 : (!evaluator, !ct) -> !ct
      %ct_19 = lattigo.ckks.add_new %evaluator, %ct_18, %ct_17 : (!evaluator, !ct, !ct) -> !ct
      %inserted_20 = tensor.insert %ct_19 into %1[%c0] : tensor<1x!ct>
      scf.yield %inserted_20 : tensor<1x!ct>
    }
    %extracted_10 = tensor.extract %3[%c0] : tensor<1x!ct>
    %ct_11 = lattigo.ckks.rotate_new %evaluator, %extracted_10, %c512 : (!evaluator, !ct, index) -> !ct
    %ct_12 = lattigo.ckks.add_new %evaluator, %extracted_10, %ct_11 : (!evaluator, !ct, !ct) -> !ct
    %inserted_13 = tensor.insert %ct_12 into %1[%c0] : tensor<1x!ct>
    return %inserted_13 : tensor<1x!ct>
  }
  func.func @matvec__encrypt__zero__f3de7246418d51b3(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %encryptor: !encryptor_pk) -> !ct attributes {client.enc_zero_func} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8192xf64>
    %pt = lattigo.ckks.new_plaintext %param : (!param) -> !pt
    %pt_0 = lattigo.ckks.encode %encoder, %cst, %pt {scale = 45 : i64} : (!encoder, tensor<8192xf64>, !pt) -> !pt
    %ct = lattigo.rlwe.encrypt %encryptor, %pt_0 : (!encryptor_pk, !pt) -> !ct
    return %ct : !ct
  }
  func.func @matvec__encrypt__arg0(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %encryptor: !encryptor_pk, %arg0: tensor<784xf32>) -> tensor<1x!ct> attributes {client.enc_func = {func_name = "matvec", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c784_i32 = arith.constant 784 : i32
    %0 = scf.for %arg1 = %c0_i32 to %c784_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<1x1024xf32>)  : i32 {
      %1 = arith.index_cast %arg1 : i32 to index
      %extracted = tensor.extract %arg0[%1] : tensor<784xf32>
      %inserted = tensor.insert %extracted into %arg2[%c0, %1] : tensor<1x1024xf32>
      scf.yield %inserted : tensor<1x1024xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt = lattigo.ckks.new_plaintext %param : (!param) -> !pt
    %pt_0 = lattigo.ckks.encode %encoder, %extracted_slice, %pt {scale = 45 : i64} : (!encoder, tensor<1024xf32>, !pt) -> !pt
    %ct = lattigo.rlwe.encrypt %encryptor, %pt_0 : (!encryptor_pk, !pt) -> !ct
    %from_elements = tensor.from_elements %ct : tensor<1x!ct>
    return %from_elements : tensor<1x!ct>
  }
  func.func @matvec__decrypt__result0(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %decryptor: !decryptor, %arg0: tensor<1x!ct>) -> tensor<512xf32> attributes {client.dec_func = {func_name = "matvec", index = 0 : i64}} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %extracted = tensor.extract %arg0[%c0] : tensor<1x!ct>
    %pt = lattigo.rlwe.decrypt %decryptor, %extracted : (!decryptor, !ct) -> !pt
    %0 = lattigo.ckks.decode %encoder, %pt, %cst : (!encoder, !pt, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %1 = scf.for %arg1 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg2 = %cst_0) -> (tensor<512xf32>)  : i32 {
      %2 = arith.remsi %arg1, %c512_i32 : i32
      %3 = arith.index_cast %arg1 : i32 to index
      %extracted_1 = tensor.extract %0[%c0, %3] : tensor<1x1024xf32>
      %4 = arith.index_cast %2 : i32 to index
      %inserted = tensor.insert %extracted_1 into %arg2[%4] : tensor<512xf32>
      scf.yield %inserted : tensor<512xf32>
    }
    return %1 : tensor<512xf32>
  }
}
