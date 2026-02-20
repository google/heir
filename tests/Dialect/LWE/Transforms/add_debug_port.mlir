// RUN: heir-opt --mlir-print-local-scope --lwe-add-debug-port=insert-debug-after-every-op=true %s | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>

!ty = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>>

func.func @simple_sum(%arg0: !ty) -> !ty {
  %c31 = arith.constant 31 : index
  %c1_i16 = arith.constant 1 : i16
  %cst = arith.constant dense<0> : tensor<32xi16>
  %inserted = tensor.insert %c1_i16 into %cst[%c31] : tensor<32xi16>
  %0 = bgv.rotate_cols %arg0 { offset = 16 } : !ty
  %1 = bgv.add %arg0, %0 : (!ty, !ty) -> !ty
  %2 = bgv.rotate_cols %1 { offset = 8 } : !ty
  %3 = bgv.add %1, %2 : (!ty, !ty) -> !ty
  %4 = bgv.rotate_cols %3 { offset = 4 } : !ty
  %5 = bgv.add %3, %4 : (!ty, !ty) -> !ty
  %6 = bgv.rotate_cols %5 { offset = 2 } : !ty
  %7 = bgv.add %5, %6 : (!ty, !ty) -> !ty
  %8 = bgv.rotate_cols %7 { offset = 1 } : !ty
  %9 = bgv.add %7, %8 : (!ty, !ty) -> !ty
  %pt = lwe.rlwe_encode %inserted {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi16> -> !pt
  %10 = bgv.mul_plain %9, %pt : (!ty, !pt) -> !ty
  %11 = bgv.rotate_cols %10 {offset = 31 : index} : !ty
  return %11 : !ty
}

// CHECK: @simple_sum
// CHECK-SAME: (%[[sk:[^:]*]]: [[sk_ty:[^,]*]],
// CHECK-SAME: %[[original_input:[^:]*]]: [[in_ty:[^)]*]])
// CHECK-SAME: -> [[out_ty:[^{]*]] {

// check calling debug func on arguments
// CHECK: call @__heir_debug
// CHECK-SAME: (%[[sk]], %[[original_input]])

// check calling debug func for each intermediate value
// CHECK-COUNT-12: call @__heir_debug
// CHECK-NOT: call @__heir_debug
