!Z1073741441_i64 = !mod_arith.int<1073741441 : i64>
!Z1073742721_i64 = !mod_arith.int<1073742721 : i64>
!Z1073742881_i64 = !mod_arith.int<1073742881 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 60>
#inverse_canonical_encoding2 = #lwe.inverse_canonical_encoding<scaling_factor = 90>
#inverse_canonical_encoding3 = #lwe.inverse_canonical_encoding<scaling_factor = 120>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 7 and 0 <= slot <= 7 }">
#modulus_chain_L4_C0 = #lwe.modulus_chain<elements = <1073742881 : i64, 1073742721 : i64, 1073741441 : i64, 1073741857 : i64, 524353 : i64>, current = 0>
#modulus_chain_L4_C1 = #lwe.modulus_chain<elements = <1073742881 : i64, 1073742721 : i64, 1073741441 : i64, 1073741857 : i64, 524353 : i64>, current = 1>
#modulus_chain_L4_C2 = #lwe.modulus_chain<elements = <1073742881 : i64, 1073742721 : i64, 1073741441 : i64, 1073741857 : i64, 524353 : i64>, current = 2>
#ring_f64_1_x8 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8>>
!rns_L0 = !rns.rns<!Z1073742881_i64>
!rns_L1 = !rns.rns<!Z1073742881_i64, !Z1073742721_i64>
!rns_L2 = !rns.rns<!Z1073742881_i64, !Z1073742721_i64, !Z1073741441_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<8xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>>
!pt1 = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding1>>
!pt2 = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding2>>
#ring_rns_L0_1_x8 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**8>>
#ring_rns_L1_1_x8 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**8>>
#ring_rns_L2_1_x8 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**8>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8, encryption_type = mix>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x8, encryption_type = mix>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x8, encryption_type = mix>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L4_C0>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L4_C1>
!ct_L1_1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding3>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L4_C1>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L4_C2>
!ct_L2_1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L4_C2>
module attributes {scheme.ckks} {
  func.func @matvec_identity__preprocessing(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>) -> (tensor<1x!pt>, tensor<1x!pt>) attributes {client.pack_func = {func_name = "matvec_identity"}} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %cst : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_1 = jaxiteword.encode %arg0, %cst_0 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %from_elements = tensor.from_elements %pt : tensor<1x!pt>
    %from_elements_2 = tensor.from_elements %pt_1 : tensor<1x!pt>
    return %from_elements, %from_elements_2 : tensor<1x!pt>, tensor<1x!pt>
  }
  func.func @matvec_identity__preprocessed(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2>, %arg3: tensor<1x!pt>, %arg4: tensor<1x!pt>) -> tensor<1x!ct_L1> attributes {client.preprocessed_func = {func_name = "matvec_identity"}} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg3[%c0] : tensor<1x!pt>
    %extracted_0 = tensor.extract %arg4[%c0] : tensor<1x!pt>
    %extracted_1 = tensor.extract %arg2[%c0] : tensor<1x!ct_L2>
    %ct = jaxiteword.rot %arg0, %extracted_1, %arg1 {index = 1 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %ct_2 = jaxiteword.mul_plain %arg0, %ct, %extracted : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_3 = jaxiteword.rot %arg0, %extracted_1, %arg1 {index = 2 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %ct_4 = jaxiteword.mul_plain %arg0, %ct_3, %extracted : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_5 = jaxiteword.mul_plain %arg0, %extracted_1, %extracted : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_6 = jaxiteword.add %arg0, %ct_5, %ct_2 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_7 = jaxiteword.add %arg0, %ct_6, %ct_4 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_8 = jaxiteword.rot %arg0, %ct_7, %arg1 {index = 3 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_9 = jaxiteword.rot %arg0, %ct_6, %arg1 {index = 6 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_10 = jaxiteword.mul_plain %arg0, %extracted_1, %extracted_0 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_11 = jaxiteword.add %arg0, %ct_10, %ct_2 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_12 = jaxiteword.add %arg0, %ct_4, %ct_8 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_13 = jaxiteword.add %arg0, %ct_12, %ct_9 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_14 = jaxiteword.add %arg0, %ct_11, %ct_13 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %0 = tensor.empty() : tensor<1x!ct_L1>
    %ct_15 = jaxiteword.mod_reduce %arg0, %ct_14 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %inserted = tensor.insert %ct_15 into %0[%c0] : tensor<1x!ct_L1>
    return %inserted : tensor<1x!ct_L1>
  }
  func.func @matvec_identity(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2> {tensor_ext.original_type = #original_type}) -> (tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}) {
    %0:2 = call @matvec_identity__preprocessing(%arg0, %arg1) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>) -> (tensor<1x!pt>, tensor<1x!pt>)
    %1 = call @matvec_identity__preprocessed(%arg0, %arg1, %arg2, %0#0, %0#1) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>, tensor<1x!ct_L2>, tensor<1x!pt>, tensor<1x!pt>) -> tensor<1x!ct_L1>
    return %1 : tensor<1x!ct_L1>
  }
  func.func @matvec_identity__encrypt__arg0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<8xf32>, %arg3: !jaxiteword.public_key<>) -> tensor<1x!ct_L2> attributes {client.enc_func = {func_name = "matvec_identity", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<1x8xf32>)  : i32 {
      %1 = arith.index_cast %arg4 : i32 to index
      %extracted = tensor.extract %arg2[%1] : tensor<8xf32>
      %inserted = tensor.insert %extracted into %arg5[%c0, %1] : tensor<1x8xf32>
      scf.yield %inserted : tensor<1x8xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %extracted_slice : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %ct = jaxiteword.encrypt %arg0, %pt, %arg3 : (!jaxiteword.crypto_context<>, !pt, !jaxiteword.public_key<>) -> !ct_L2
    %from_elements = tensor.from_elements %ct : tensor<1x!ct_L2>
    return %from_elements : tensor<1x!ct_L2>
  }
  func.func @matvec_identity__decrypt__result0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L1>, %arg3: !jaxiteword.private_key<>) -> tensor<8xf32> attributes {client.dec_func = {func_name = "matvec_identity", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L1>
    %pt = jaxiteword.decrypt %arg0, %extracted, %arg3 : (!jaxiteword.crypto_context<>, !ct_L1, !jaxiteword.private_key<>) -> !pt1
    %0 = jaxiteword.decode %arg0, %pt : (!jaxiteword.crypto_context<>, !pt1) -> tensor<1x8xf32>
    %1 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<8xf32>)  : i32 {
      %2 = arith.subi %c7_i32, %arg4 : i32
      %3 = arith.index_cast %2 : i32 to index
      %extracted_0 = tensor.extract %0[%c0, %3] : tensor<1x8xf32>
      %inserted = tensor.insert %extracted_0 into %arg5[%3] : tensor<8xf32>
      scf.yield %inserted : tensor<8xf32>
    }
    return %1 : tensor<8xf32>
  }
  func.func @matvec_shift__preprocessing(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>) -> (tensor<1x!pt>, tensor<1x!pt>) attributes {client.pack_func = {func_name = "matvec_shift"}} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %cst : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_1 = jaxiteword.encode %arg0, %cst_0 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %from_elements = tensor.from_elements %pt : tensor<1x!pt>
    %from_elements_2 = tensor.from_elements %pt_1 : tensor<1x!pt>
    return %from_elements, %from_elements_2 : tensor<1x!pt>, tensor<1x!pt>
  }
  func.func @matvec_shift__preprocessed(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2>, %arg3: tensor<1x!pt>, %arg4: tensor<1x!pt>) -> tensor<1x!ct_L1> attributes {client.preprocessed_func = {func_name = "matvec_shift"}} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg3[%c0] : tensor<1x!pt>
    %extracted_0 = tensor.extract %arg4[%c0] : tensor<1x!pt>
    %extracted_1 = tensor.extract %arg2[%c0] : tensor<1x!ct_L2>
    %ct = jaxiteword.mul_plain %arg0, %extracted_1, %extracted : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_2 = jaxiteword.rot %arg0, %extracted_1, %arg1 {index = 1 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %ct_3 = jaxiteword.rot %arg0, %extracted_1, %arg1 {index = 2 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %ct_4 = jaxiteword.mul_plain %arg0, %ct_3, %extracted : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_5 = jaxiteword.mul_plain %arg0, %ct_2, %extracted : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_6 = jaxiteword.add %arg0, %ct, %ct_5 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_7 = jaxiteword.add %arg0, %ct_6, %ct_4 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_8 = jaxiteword.rot %arg0, %ct_7, %arg1 {index = 3 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_9 = jaxiteword.rot %arg0, %ct_6, %arg1 {index = 6 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_10 = jaxiteword.mul_plain %arg0, %ct_2, %extracted_0 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_11 = jaxiteword.add %arg0, %ct, %ct_10 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_12 = jaxiteword.add %arg0, %ct_4, %ct_8 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_13 = jaxiteword.add %arg0, %ct_12, %ct_9 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_14 = jaxiteword.add %arg0, %ct_11, %ct_13 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %0 = tensor.empty() : tensor<1x!ct_L1>
    %ct_15 = jaxiteword.mod_reduce %arg0, %ct_14 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %inserted = tensor.insert %ct_15 into %0[%c0] : tensor<1x!ct_L1>
    return %inserted : tensor<1x!ct_L1>
  }
  func.func @matvec_shift(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2> {tensor_ext.original_type = #original_type}) -> (tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}) {
    %0:2 = call @matvec_shift__preprocessing(%arg0, %arg1) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>) -> (tensor<1x!pt>, tensor<1x!pt>)
    %1 = call @matvec_shift__preprocessed(%arg0, %arg1, %arg2, %0#0, %0#1) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>, tensor<1x!ct_L2>, tensor<1x!pt>, tensor<1x!pt>) -> tensor<1x!ct_L1>
    return %1 : tensor<1x!ct_L1>
  }
  func.func @matvec_shift__encrypt__arg0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<8xf32>, %arg3: !jaxiteword.public_key<>) -> tensor<1x!ct_L2> attributes {client.enc_func = {func_name = "matvec_shift", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<1x8xf32>)  : i32 {
      %1 = arith.index_cast %arg4 : i32 to index
      %extracted = tensor.extract %arg2[%1] : tensor<8xf32>
      %inserted = tensor.insert %extracted into %arg5[%c0, %1] : tensor<1x8xf32>
      scf.yield %inserted : tensor<1x8xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %extracted_slice : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %ct = jaxiteword.encrypt %arg0, %pt, %arg3 : (!jaxiteword.crypto_context<>, !pt, !jaxiteword.public_key<>) -> !ct_L2
    %from_elements = tensor.from_elements %ct : tensor<1x!ct_L2>
    return %from_elements : tensor<1x!ct_L2>
  }
  func.func @matvec_shift__decrypt__result0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L1>, %arg3: !jaxiteword.private_key<>) -> tensor<8xf32> attributes {client.dec_func = {func_name = "matvec_shift", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L1>
    %pt = jaxiteword.decrypt %arg0, %extracted, %arg3 : (!jaxiteword.crypto_context<>, !ct_L1, !jaxiteword.private_key<>) -> !pt1
    %0 = jaxiteword.decode %arg0, %pt : (!jaxiteword.crypto_context<>, !pt1) -> tensor<1x8xf32>
    %1 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<8xf32>)  : i32 {
      %2 = arith.subi %c7_i32, %arg4 : i32
      %3 = arith.index_cast %2 : i32 to index
      %extracted_0 = tensor.extract %0[%c0, %3] : tensor<1x8xf32>
      %inserted = tensor.insert %extracted_0 into %arg5[%3] : tensor<8xf32>
      scf.yield %inserted : tensor<8xf32>
    }
    return %1 : tensor<8xf32>
  }
  func.func @matvec_random__preprocessing(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>) -> (tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>) attributes {client.pack_func = {func_name = "matvec_random"}} {
    %cst = arith.constant dense<[0.811626255, 1.44533789, 0.920695543, 1.07704544, 0.678766131, 1.3587923, 1.236010e+00, 0.777831316]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[1.90635717, 0.139110535, 0.653335392, 1.22558773, 0.285577029, 6.922510e-01, 1.85156107, 0.268135756]> : tensor<8xf32>
    %cst_1 = arith.constant dense<[1.49078846, 1.94282877, 1.26252055, 0.188255787, 1.40004277, 1.08812928, 1.13874948, 0.472367436]> : tensor<8xf32>
    %cst_2 = arith.constant dense<[0.331872642, 0.451223463, 0.185931846, 1.23745108, 1.68164098, 0.365038335, 1.25433517, 0.936289727]> : tensor<8xf32>
    %cst_3 = arith.constant dense<[1.0408361, 1.94221079, 0.718127608, 0.39643541, 0.503444314, 0.655074835, 0.423995823, 0.223598033]> : tensor<8xf32>
    %cst_4 = arith.constant dense<[0.165338188, 1.57275236, 0.83848685, 0.396389604, 0.445467442, 0.796087503, 0.966532945, 1.90288258]> : tensor<8xf32>
    %cst_5 = arith.constant dense<[0.678060233, 1.59183431, 1.93470085, 1.82770872, 1.88504803, 0.615563154, 0.210358858, 0.448468566]> : tensor<8xf32>
    %cst_6 = arith.constant dense<[1.0970372, 0.47938019, 1.63595498, 0.591681957, 1.80017197, 1.67460132, 1.74573469, 1.24211848]> : tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %cst : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_7 = jaxiteword.encode %arg0, %cst_0 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_8 = jaxiteword.encode %arg0, %cst_1 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_9 = jaxiteword.encode %arg0, %cst_2 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_10 = jaxiteword.encode %arg0, %cst_3 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_11 = jaxiteword.encode %arg0, %cst_4 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_12 = jaxiteword.encode %arg0, %cst_5 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_13 = jaxiteword.encode %arg0, %cst_6 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %from_elements = tensor.from_elements %pt : tensor<1x!pt>
    %from_elements_14 = tensor.from_elements %pt_7 : tensor<1x!pt>
    %from_elements_15 = tensor.from_elements %pt_8 : tensor<1x!pt>
    %from_elements_16 = tensor.from_elements %pt_9 : tensor<1x!pt>
    %from_elements_17 = tensor.from_elements %pt_10 : tensor<1x!pt>
    %from_elements_18 = tensor.from_elements %pt_11 : tensor<1x!pt>
    %from_elements_19 = tensor.from_elements %pt_12 : tensor<1x!pt>
    %from_elements_20 = tensor.from_elements %pt_13 : tensor<1x!pt>
    return %from_elements, %from_elements_14, %from_elements_15, %from_elements_16, %from_elements_17, %from_elements_18, %from_elements_19, %from_elements_20 : tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>
  }
  func.func @matvec_random__preprocessed(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2>, %arg3: tensor<1x!pt>, %arg4: tensor<1x!pt>, %arg5: tensor<1x!pt>, %arg6: tensor<1x!pt>, %arg7: tensor<1x!pt>, %arg8: tensor<1x!pt>, %arg9: tensor<1x!pt>, %arg10: tensor<1x!pt>) -> tensor<1x!ct_L1> attributes {client.preprocessed_func = {func_name = "matvec_random"}} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg3[%c0] : tensor<1x!pt>
    %extracted_0 = tensor.extract %arg4[%c0] : tensor<1x!pt>
    %extracted_1 = tensor.extract %arg5[%c0] : tensor<1x!pt>
    %extracted_2 = tensor.extract %arg6[%c0] : tensor<1x!pt>
    %extracted_3 = tensor.extract %arg7[%c0] : tensor<1x!pt>
    %extracted_4 = tensor.extract %arg8[%c0] : tensor<1x!pt>
    %extracted_5 = tensor.extract %arg9[%c0] : tensor<1x!pt>
    %extracted_6 = tensor.extract %arg10[%c0] : tensor<1x!pt>
    %extracted_7 = tensor.extract %arg2[%c0] : tensor<1x!ct_L2>
    %ct = jaxiteword.mul_plain %arg0, %extracted_7, %extracted : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_8 = jaxiteword.rot %arg0, %extracted_7, %arg1 {index = 1 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %ct_9 = jaxiteword.mul_plain %arg0, %ct_8, %extracted_0 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_10 = jaxiteword.rot %arg0, %extracted_7, %arg1 {index = 2 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %ct_11 = jaxiteword.mul_plain %arg0, %ct_10, %extracted_1 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_12 = jaxiteword.mul_plain %arg0, %extracted_7, %extracted_2 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_13 = jaxiteword.mul_plain %arg0, %ct_8, %extracted_3 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_14 = jaxiteword.mul_plain %arg0, %ct_10, %extracted_4 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_15 = jaxiteword.add %arg0, %ct_12, %ct_13 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_16 = jaxiteword.add %arg0, %ct_15, %ct_14 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_17 = jaxiteword.rot %arg0, %ct_16, %arg1 {index = 3 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_18 = jaxiteword.mul_plain %arg0, %extracted_7, %extracted_5 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_19 = jaxiteword.mul_plain %arg0, %ct_8, %extracted_6 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_20 = jaxiteword.add %arg0, %ct_18, %ct_19 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_21 = jaxiteword.rot %arg0, %ct_20, %arg1 {index = 6 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_22 = jaxiteword.add %arg0, %ct, %ct_9 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_23 = jaxiteword.add %arg0, %ct_11, %ct_17 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_24 = jaxiteword.add %arg0, %ct_23, %ct_21 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_25 = jaxiteword.add %arg0, %ct_22, %ct_24 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %0 = tensor.empty() : tensor<1x!ct_L1>
    %ct_26 = jaxiteword.mod_reduce %arg0, %ct_25 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %inserted = tensor.insert %ct_26 into %0[%c0] : tensor<1x!ct_L1>
    return %inserted : tensor<1x!ct_L1>
  }
  func.func @matvec_random(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2> {tensor_ext.original_type = #original_type}) -> (tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}) {
    %0:8 = call @matvec_random__preprocessing(%arg0, %arg1) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>) -> (tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>)
    %1 = call @matvec_random__preprocessed(%arg0, %arg1, %arg2, %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>, tensor<1x!ct_L2>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>) -> tensor<1x!ct_L1>
    return %1 : tensor<1x!ct_L1>
  }
  func.func @matvec_random__encrypt__arg0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<8xf32>, %arg3: !jaxiteword.public_key<>) -> tensor<1x!ct_L2> attributes {client.enc_func = {func_name = "matvec_random", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<1x8xf32>)  : i32 {
      %1 = arith.index_cast %arg4 : i32 to index
      %extracted = tensor.extract %arg2[%1] : tensor<8xf32>
      %inserted = tensor.insert %extracted into %arg5[%c0, %1] : tensor<1x8xf32>
      scf.yield %inserted : tensor<1x8xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %extracted_slice : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %ct = jaxiteword.encrypt %arg0, %pt, %arg3 : (!jaxiteword.crypto_context<>, !pt, !jaxiteword.public_key<>) -> !ct_L2
    %from_elements = tensor.from_elements %ct : tensor<1x!ct_L2>
    return %from_elements : tensor<1x!ct_L2>
  }
  func.func @matvec_random__decrypt__result0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L1>, %arg3: !jaxiteword.private_key<>) -> tensor<8xf32> attributes {client.dec_func = {func_name = "matvec_random", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L1>
    %pt = jaxiteword.decrypt %arg0, %extracted, %arg3 : (!jaxiteword.crypto_context<>, !ct_L1, !jaxiteword.private_key<>) -> !pt1
    %0 = jaxiteword.decode %arg0, %pt : (!jaxiteword.crypto_context<>, !pt1) -> tensor<1x8xf32>
    %1 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<8xf32>)  : i32 {
      %2 = arith.subi %c7_i32, %arg4 : i32
      %3 = arith.index_cast %2 : i32 to index
      %extracted_0 = tensor.extract %0[%c0, %3] : tensor<1x8xf32>
      %inserted = tensor.insert %extracted_0 into %arg5[%3] : tensor<8xf32>
      scf.yield %inserted : tensor<8xf32>
    }
    return %1 : tensor<8xf32>
  }
  func.func @matvec_chain__preprocessing(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>) -> (tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<2x!pt1>, tensor<2x!pt1>, tensor<2x!pt1>, tensor<2x!pt1>) attributes {client.pack_func = {func_name = "matvec_chain"}} {
    %cst = arith.constant dense<[1.340000e+00, 1.220000e+00, 1.050000e+00, 1.500000e+00, 1.010000e+00, 0.879999995, 5.000000e-01, 1.060000e+00]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[5.800000e-01, 5.200000e-01, 0.889999985, 8.600000e-01, 1.170000e+00, 0.819999992, 1.490000e+00, 1.410000e+00]> : tensor<8xf32>
    %cst_1 = arith.constant dense<[1.260000e+00, 1.090000e+00, 1.430000e+00, 1.260000e+00, 6.100000e-01, 8.500000e-01, 6.700000e-01, 0.709999978]> : tensor<8xf32>
    %cst_2 = arith.constant dense<[0.819999992, 1.330000e+00, 7.900000e-01, 7.400000e-01, 1.060000e+00, 1.340000e+00, 1.090000e+00, 6.300000e-01]> : tensor<8xf32>
    %cst_3 = arith.constant dense<[1.160000e+00, 0.839999973, 1.020000e+00, 0.689999997, 6.600000e-01, 8.600000e-01, 1.190000e+00, 6.500000e-01]> : tensor<8xf32>
    %cst_4 = arith.constant dense<[1.350000e+00, 1.050000e+00, 1.400000e+00, 1.070000e+00, 6.500000e-01, 5.400000e-01, 8.000000e-01, 0.899999976]> : tensor<8xf32>
    %cst_5 = arith.constant dense<[0.819999992, 0.899999976, 7.400000e-01, 1.050000e+00, 1.080000e+00, 1.480000e+00, 6.000000e-01, 1.200000e+00]> : tensor<8xf32>
    %cst_6 = arith.constant dense<[1.190000e+00, 1.200000e+00, 0.839999973, 1.350000e+00, 1.020000e+00, 7.600000e-01, 1.390000e+00, 1.130000e+00]> : tensor<8xf32>
    %cst_7 = arith.constant dense<[1.200000e+00, 0.889999985, 1.030000e+00, 7.300000e-01, 9.300000e-01, 7.500000e-01, 0.839999973, 1.170000e+00]> : tensor<8xf32>
    %cst_8 = arith.constant dense<[7.900000e-01, 0.839999973, 1.030000e+00, 7.900000e-01, 1.390000e+00, 9.800000e-01, 8.000000e-01, 9.200000e-01]> : tensor<8xf32>
    %cst_9 = arith.constant dense<[7.300000e-01, 1.230000e+00, 1.130000e+00, 1.130000e+00, 1.440000e+00, 1.490000e+00, 1.020000e+00, 1.180000e+00]> : tensor<8xf32>
    %cst_10 = arith.constant dense<[1.120000e+00, 1.110000e+00, 1.380000e+00, 1.050000e+00, 0.939999997, 1.350000e+00, 5.900000e-01, 1.000000e+00]> : tensor<8xf32>
    %cst_11 = arith.constant dense<[6.200000e-01, 6.200000e-01, 1.010000e+00, 1.220000e+00, 5.600000e-01, 1.220000e+00, 9.300000e-01, 9.300000e-01]> : tensor<8xf32>
    %cst_12 = arith.constant dense<[0.819999992, 1.330000e+00, 1.170000e+00, 9.200000e-01, 0.899999976, 1.110000e+00, 1.220000e+00, 9.900000e-01]> : tensor<8xf32>
    %cst_13 = arith.constant dense<[6.800000e-01, 0.819999992, 9.300000e-01, 9.100000e-01, 1.100000e+00, 1.090000e+00, 1.480000e+00, 1.240000e+00]> : tensor<8xf32>
    %cst_14 = arith.constant dense<[6.800000e-01, 8.600000e-01, 8.100000e-01, 1.370000e+00, 1.050000e+00, 1.120000e+00, 1.180000e+00, 9.800000e-01]> : tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %cst : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_15 = jaxiteword.encode %arg0, %cst_0 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_16 = jaxiteword.encode %arg0, %cst_1 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_17 = jaxiteword.encode %arg0, %cst_2 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_18 = jaxiteword.encode %arg0, %cst_3 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_19 = jaxiteword.encode %arg0, %cst_4 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_20 = jaxiteword.encode %arg0, %cst_5 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_21 = jaxiteword.encode %arg0, %cst_6 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %pt_22 = jaxiteword.encode %arg0, %cst_7 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt1
    %pt_23 = jaxiteword.encode %arg0, %cst_8 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt1
    %pt_24 = jaxiteword.encode %arg0, %cst_9 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt1
    %pt_25 = jaxiteword.encode %arg0, %cst_10 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt1
    %pt_26 = jaxiteword.encode %arg0, %cst_11 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt1
    %pt_27 = jaxiteword.encode %arg0, %cst_12 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt1
    %pt_28 = jaxiteword.encode %arg0, %cst_13 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt1
    %pt_29 = jaxiteword.encode %arg0, %cst_14 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt1
    %from_elements = tensor.from_elements %pt : tensor<1x!pt>
    %from_elements_30 = tensor.from_elements %pt_15 : tensor<1x!pt>
    %from_elements_31 = tensor.from_elements %pt_16 : tensor<1x!pt>
    %from_elements_32 = tensor.from_elements %pt_17 : tensor<1x!pt>
    %from_elements_33 = tensor.from_elements %pt_18 : tensor<1x!pt>
    %from_elements_34 = tensor.from_elements %pt_19 : tensor<1x!pt>
    %from_elements_35 = tensor.from_elements %pt_20 : tensor<1x!pt>
    %from_elements_36 = tensor.from_elements %pt_21 : tensor<1x!pt>
    %from_elements_37 = tensor.from_elements %pt_22, %pt_23 : tensor<2x!pt1>
    %from_elements_38 = tensor.from_elements %pt_24, %pt_25 : tensor<2x!pt1>
    %from_elements_39 = tensor.from_elements %pt_26, %pt_27 : tensor<2x!pt1>
    %from_elements_40 = tensor.from_elements %pt_28, %pt_29 : tensor<2x!pt1>
    return %from_elements, %from_elements_30, %from_elements_31, %from_elements_32, %from_elements_33, %from_elements_34, %from_elements_35, %from_elements_36, %from_elements_37, %from_elements_38, %from_elements_39, %from_elements_40 : tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<2x!pt1>, tensor<2x!pt1>, tensor<2x!pt1>, tensor<2x!pt1>
  }
  func.func @matvec_chain__preprocessed(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2>, %arg3: tensor<1x!pt>, %arg4: tensor<1x!pt>, %arg5: tensor<1x!pt>, %arg6: tensor<1x!pt>, %arg7: tensor<1x!pt>, %arg8: tensor<1x!pt>, %arg9: tensor<1x!pt>, %arg10: tensor<1x!pt>, %arg11: tensor<2x!pt1>, %arg12: tensor<2x!pt1>, %arg13: tensor<2x!pt1>, %arg14: tensor<2x!pt1>) -> tensor<1x!ct_L0> attributes {client.preprocessed_func = {func_name = "matvec_chain"}} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg3[%c0] : tensor<1x!pt>
    %extracted_0 = tensor.extract %arg4[%c0] : tensor<1x!pt>
    %extracted_1 = tensor.extract %arg5[%c0] : tensor<1x!pt>
    %extracted_2 = tensor.extract %arg6[%c0] : tensor<1x!pt>
    %extracted_3 = tensor.extract %arg7[%c0] : tensor<1x!pt>
    %extracted_4 = tensor.extract %arg8[%c0] : tensor<1x!pt>
    %extracted_5 = tensor.extract %arg9[%c0] : tensor<1x!pt>
    %extracted_6 = tensor.extract %arg10[%c0] : tensor<1x!pt>
    %extracted_7 = tensor.extract %arg11[%c0] : tensor<2x!pt1>
    %extracted_8 = tensor.extract %arg11[%c1] : tensor<2x!pt1>
    %extracted_9 = tensor.extract %arg12[%c0] : tensor<2x!pt1>
    %extracted_10 = tensor.extract %arg12[%c1] : tensor<2x!pt1>
    %extracted_11 = tensor.extract %arg13[%c0] : tensor<2x!pt1>
    %extracted_12 = tensor.extract %arg13[%c1] : tensor<2x!pt1>
    %extracted_13 = tensor.extract %arg14[%c0] : tensor<2x!pt1>
    %extracted_14 = tensor.extract %arg14[%c1] : tensor<2x!pt1>
    %extracted_15 = tensor.extract %arg2[%c0] : tensor<1x!ct_L2>
    %ct = jaxiteword.mul_plain %arg0, %extracted_15, %extracted : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_16 = jaxiteword.rot %arg0, %extracted_15, %arg1 {index = 1 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %ct_17 = jaxiteword.mul_plain %arg0, %ct_16, %extracted_0 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_18 = jaxiteword.rot %arg0, %extracted_15, %arg1 {index = 2 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %ct_19 = jaxiteword.mul_plain %arg0, %ct_18, %extracted_1 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_20 = jaxiteword.mul_plain %arg0, %extracted_15, %extracted_2 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_21 = jaxiteword.mul_plain %arg0, %ct_16, %extracted_3 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_22 = jaxiteword.mul_plain %arg0, %ct_18, %extracted_4 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_23 = jaxiteword.add %arg0, %ct_20, %ct_21 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_24 = jaxiteword.add %arg0, %ct_23, %ct_22 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_25 = jaxiteword.rot %arg0, %ct_24, %arg1 {index = 3 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_26 = jaxiteword.mul_plain %arg0, %extracted_15, %extracted_5 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_27 = jaxiteword.mul_plain %arg0, %ct_16, %extracted_6 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_28 = jaxiteword.add %arg0, %ct_26, %ct_27 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_29 = jaxiteword.rot %arg0, %ct_28, %arg1 {index = 6 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_30 = jaxiteword.add %arg0, %ct, %ct_17 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_31 = jaxiteword.add %arg0, %ct_19, %ct_25 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_32 = jaxiteword.add %arg0, %ct_31, %ct_29 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_33 = jaxiteword.add %arg0, %ct_30, %ct_32 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_34 = jaxiteword.mod_reduce %arg0, %ct_33 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %ct_35 = jaxiteword.mul_plain %arg0, %ct_34, %extracted_7 : (!jaxiteword.crypto_context<>, !ct_L1, !pt1) -> !ct_L1_1
    %ct_36 = jaxiteword.rot %arg0, %ct_33, %arg1 {index = 1 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_37 = jaxiteword.mod_reduce %arg0, %ct_36 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %ct_38 = jaxiteword.mul_plain %arg0, %ct_37, %extracted_8 : (!jaxiteword.crypto_context<>, !ct_L1, !pt1) -> !ct_L1_1
    %ct_39 = jaxiteword.rot %arg0, %ct_33, %arg1 {index = 2 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_40 = jaxiteword.mod_reduce %arg0, %ct_39 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %ct_41 = jaxiteword.mul_plain %arg0, %ct_40, %extracted_9 : (!jaxiteword.crypto_context<>, !ct_L1, !pt1) -> !ct_L1_1
    %ct_42 = jaxiteword.mul_plain %arg0, %ct_34, %extracted_10 : (!jaxiteword.crypto_context<>, !ct_L1, !pt1) -> !ct_L1_1
    %ct_43 = jaxiteword.mul_plain %arg0, %ct_37, %extracted_11 : (!jaxiteword.crypto_context<>, !ct_L1, !pt1) -> !ct_L1_1
    %ct_44 = jaxiteword.mul_plain %arg0, %ct_40, %extracted_12 : (!jaxiteword.crypto_context<>, !ct_L1, !pt1) -> !ct_L1_1
    %ct_45 = jaxiteword.add %arg0, %ct_42, %ct_43 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_46 = jaxiteword.add %arg0, %ct_45, %ct_44 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_47 = jaxiteword.rot %arg0, %ct_46, %arg1 {index = 3 : i64} : (!jaxiteword.crypto_context<>, !ct_L1_1, !jaxiteword.eval_key<>) -> !ct_L1_1
    %ct_48 = jaxiteword.mul_plain %arg0, %ct_34, %extracted_13 : (!jaxiteword.crypto_context<>, !ct_L1, !pt1) -> !ct_L1_1
    %ct_49 = jaxiteword.mul_plain %arg0, %ct_37, %extracted_14 : (!jaxiteword.crypto_context<>, !ct_L1, !pt1) -> !ct_L1_1
    %ct_50 = jaxiteword.add %arg0, %ct_48, %ct_49 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_51 = jaxiteword.rot %arg0, %ct_50, %arg1 {index = 6 : i64} : (!jaxiteword.crypto_context<>, !ct_L1_1, !jaxiteword.eval_key<>) -> !ct_L1_1
    %ct_52 = jaxiteword.add %arg0, %ct_35, %ct_38 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_53 = jaxiteword.add %arg0, %ct_41, %ct_47 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_54 = jaxiteword.add %arg0, %ct_53, %ct_51 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_55 = jaxiteword.add %arg0, %ct_52, %ct_54 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %0 = tensor.empty() : tensor<1x!ct_L0>
    %ct_56 = jaxiteword.mod_reduce %arg0, %ct_55 : (!jaxiteword.crypto_context<>, !ct_L1_1) -> !ct_L0
    %inserted = tensor.insert %ct_56 into %0[%c0] : tensor<1x!ct_L0>
    return %inserted : tensor<1x!ct_L0>
  }
  func.func @matvec_chain(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2> {tensor_ext.original_type = #original_type}) -> (tensor<1x!ct_L0> {tensor_ext.original_type = #original_type}) {
    %0:12 = call @matvec_chain__preprocessing(%arg0, %arg1) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>) -> (tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<2x!pt1>, tensor<2x!pt1>, tensor<2x!pt1>, tensor<2x!pt1>)
    %1 = call @matvec_chain__preprocessed(%arg0, %arg1, %arg2, %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7, %0#8, %0#9, %0#10, %0#11) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>, tensor<1x!ct_L2>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<1x!pt>, tensor<2x!pt1>, tensor<2x!pt1>, tensor<2x!pt1>, tensor<2x!pt1>) -> tensor<1x!ct_L0>
    return %1 : tensor<1x!ct_L0>
  }
  func.func @matvec_chain__encrypt__arg0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<8xf32>, %arg3: !jaxiteword.public_key<>) -> tensor<1x!ct_L2> attributes {client.enc_func = {func_name = "matvec_chain", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<1x8xf32>)  : i32 {
      %1 = arith.index_cast %arg4 : i32 to index
      %extracted = tensor.extract %arg2[%1] : tensor<8xf32>
      %inserted = tensor.insert %extracted into %arg5[%c0, %1] : tensor<1x8xf32>
      scf.yield %inserted : tensor<1x8xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %extracted_slice : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %ct = jaxiteword.encrypt %arg0, %pt, %arg3 : (!jaxiteword.crypto_context<>, !pt, !jaxiteword.public_key<>) -> !ct_L2
    %from_elements = tensor.from_elements %ct : tensor<1x!ct_L2>
    return %from_elements : tensor<1x!ct_L2>
  }
  func.func @matvec_chain__decrypt__result0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L0>, %arg3: !jaxiteword.private_key<>) -> tensor<8xf32> attributes {client.dec_func = {func_name = "matvec_chain", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L0>
    %pt = jaxiteword.decrypt %arg0, %extracted, %arg3 : (!jaxiteword.crypto_context<>, !ct_L0, !jaxiteword.private_key<>) -> !pt2
    %0 = jaxiteword.decode %arg0, %pt : (!jaxiteword.crypto_context<>, !pt2) -> tensor<1x8xf32>
    %1 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<8xf32>)  : i32 {
      %2 = arith.subi %c7_i32, %arg4 : i32
      %3 = arith.index_cast %2 : i32 to index
      %extracted_0 = tensor.extract %0[%c0, %3] : tensor<1x8xf32>
      %inserted = tensor.insert %extracted_0 into %arg5[%3] : tensor<8xf32>
      scf.yield %inserted : tensor<8xf32>
    }
    return %1 : tensor<8xf32>
  }
  func.func @matvec_identity__generate_crypto_context(%arg0: !jaxiteword.public_key<>, %arg1: !jaxiteword.private_key<>, %arg2: !jaxiteword.eval_key<>) -> !jaxiteword.crypto_context<> {
    %0 = jaxiteword.gen_params %arg0, %arg1, %arg2 {batch = 1 : i32, c = 4 : i32, compositeDegree = 1 : i32, degree = 16 : i64, dnum = 3 : i32, numEvalMult = 1 : i32, numSlots = 8 : i64, pTowers = array<i64: 1073740609, 1073739937, 1073739649>, qTowers = array<i64: 1073742881, 1073742721, 1073741441, 1073741857, 524353>, r = 4 : i32, scalingFactor = 0x42C0000000000000 : f64} : (!jaxiteword.public_key<>, !jaxiteword.private_key<>, !jaxiteword.eval_key<>) -> !jaxiteword.crypto_context<>
    return %0 : !jaxiteword.crypto_context<>
  }
  func.func @matvec_identity__configure_crypto_context(%arg0: !jaxiteword.crypto_context<>) {
    jaxiteword.program_initialization %arg0 {batch = 1 : i32, c = 4 : i32, dnum = 3 : i32, r = 4 : i32, totalHemulLevels = 1 : i64, totalRotationIndices = array<i64: 1, 2, 3, 6>} : (!jaxiteword.crypto_context<>) -> ()
    return
  }
}
