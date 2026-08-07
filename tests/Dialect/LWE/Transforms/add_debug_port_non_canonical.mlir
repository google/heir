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
#ciphertext_space_L0_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb, size = 3>

!ct_ty = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!ct_sq_ty = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_D3_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

func.func @mul_relin(%arg0: !ct_ty, %arg1: !ct_ty) -> !ct_ty {
  %0 = bgv.mul %arg0, %arg1 : (!ct_ty, !ct_ty) -> !ct_sq_ty
  %1 = bgv.relinearize %0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : !ct_sq_ty -> !ct_ty
  return %1 : !ct_ty
}

// CHECK: @mul_relin

// Both canonical arguments are debugged.
// CHECK: call @__heir_debug
// CHECK: call @__heir_debug

// The dimension-3 product is not: bgv.mul is followed directly by the
// relinearize, with no intervening debug call.
// CHECK: bgv.mul
// CHECK-NOT: call @__heir_debug
// CHECK: bgv.relinearize

// The relinearized (canonical) result is debugged again.
// CHECK: call @__heir_debug
// CHECK-NOT: call @__heir_debug
