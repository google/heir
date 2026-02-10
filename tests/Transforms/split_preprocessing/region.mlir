// RUN: heir-opt --split-preprocessing %s | FileCheck %s

// Tests that a region (affine.for) can be moved into the preprocessing function,
// and the %cst used in the region is also backtracked in the dataflow.

// CHECK: ![[pt:.*]] = !lwe.lwe_plaintext

// CHECK: func.func @region__preprocessing() -> ![[pt]]
// CHECK: %[[cst:.*]] arith.constant dense_resource
// CHECK: affine.for
// CHECK: call @_assign_layout
// CHECK: return

// CHECK: func.func @region__preprocessed(
// CHECK-SAME: %[[arg0:.*]]: tensor<1x![[ct_L2:.*]]>,
// CHECK-SAME: %[[pt:.*]]: ![[pt]])

!Z35184371138561_i64 = !mod_arith.int<35184371138561 : i64>
!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797017456641_i64 = !mod_arith.int<36028797017456641 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (-28i2 - i3 + slot) mod 1024 = 0 and 0 <= i1 <= 5 and 0 <= i2 <= 27 and 0 <= i3 <= 27 and 0 <= slot <= 1023 }">
#modulus_chain_L2_C2 = #lwe.modulus_chain<elements = <36028797017456641 : i64, 35184371138561 : i64, 35184372121601 : i64>, current = 2>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L2 = !rns.rns<!Z36028797017456641_i64, !Z35184371138561_i64, !Z35184372121601_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<1x6x28x28xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L2_1_x1024 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x1024, encryption_type = mix>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L2_C2>
module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184371138561, 35184372121601], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 45>, scheme.ckks} {
  func.func private @_assign_layout_1845394520611349023(%arg0: tensor<28x28xf32>) -> tensor<1x1024xf32> attributes {client.pack_func = {func_name = "region"}} {
    %c0 = arith.constant 0 : index
    %c28_i32 = arith.constant 28 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c784_i32 = arith.constant 784 : i32
    %0 = scf.for %arg1 = %c0_i32 to %c784_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<1x1024xf32>)  : i32 {
      %1 = arith.divsi %arg1, %c28_i32 : i32
      %2 = arith.remsi %arg1, %c28_i32 : i32
      %3 = arith.index_cast %1 : i32 to index
      %4 = arith.index_cast %2 : i32 to index
      %extracted = tensor.extract %arg0[%3, %4] : tensor<28x28xf32>
      %5 = arith.index_cast %arg1 : i32 to index
      %inserted = tensor.insert %extracted into %arg2[%c0, %5] : tensor<1x1024xf32>
      scf.yield %inserted : tensor<1x1024xf32>
    }
    return %0 : tensor<1x1024xf32>
  }
  func.func @region(%arg0: tensor<1x!ct_L2> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x1x32x32xf32>, layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-32i2 - i3 + slot) mod 1024 = 0 and 0 <= i2 <= 31 and 0 <= i3 <= 31 and 0 <= slot <= 1023 }">>}) -> (tensor<1x!ct_L2> {tensor_ext.original_type = #original_type}) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense_resource<torch_tensor_6_torch.float32> : tensor<6xf32>
    %0 = tensor.empty() : tensor<1x6x28x28xf32>
    %1 = affine.for %arg1 = 0 to 6 iter_args(%arg2 = %0) -> (tensor<1x6x28x28xf32>) {
      %4 = affine.for %arg3 = 0 to 28 iter_args(%arg4 = %arg2) -> (tensor<1x6x28x28xf32>) {
        %5 = affine.for %arg5 = 0 to 28 iter_args(%arg6 = %arg4) -> (tensor<1x6x28x28xf32>) {
          %extracted = tensor.extract %cst[%arg1] : tensor<6xf32>
          %inserted = tensor.insert %extracted into %arg6[%c0, %arg1, %arg3, %arg5] : tensor<1x6x28x28xf32>
          affine.yield %inserted : tensor<1x6x28x28xf32>
        }
        affine.yield %5 : tensor<1x6x28x28xf32>
      }
      affine.yield %4 : tensor<1x6x28x28xf32>
    }
    %extracted_slice = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 28, 28] [1, 1, 1, 1] : tensor<1x6x28x28xf32> to tensor<28x28xf32>
    %2 = call @_assign_layout_1845394520611349023(%extracted_slice) : (tensor<28x28xf32>) -> tensor<1x1024xf32>
    %extracted_slice_0 = tensor.extract_slice %2[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt = lwe.rlwe_encode %extracted_slice_0 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements = tensor.from_elements %pt : tensor<1x!pt>
    %3 = ckks.add_plain %arg0, %from_elements : (tensor<1x!ct_L2>, tensor<1x!pt>) -> tensor<1x!ct_L2>
    return %3 : tensor<1x!ct_L2>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_6_torch.float32: "0x0400000008B030BEDAE5A43C7D68BF3D7881233ECDE4E13DE065E2BD"
    }
  }
#-}
