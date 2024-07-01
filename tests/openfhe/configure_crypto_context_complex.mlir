// RUN: heir-opt --openfhe-configure-crypto-context=entry-function=complex_func %s | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>
#ideal = #polynomial.int_polynomial<1 + x**32>
#ring= #polynomial.ring<coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus=#ideal>
#params = #lwe.rlwe_params<ring=#ring>
!in_ty = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type = tensor<32xi16>>
!out_ty = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type = i16>
!ctxt_ty = !openfhe.crypto_context
!plain_ty = !lwe.rlwe_plaintext<encoding = #encoding, ring = #ring, underlying_type = tensor<32xi16>>

func.func @complex_func(%arg0: !ctxt_ty, %arg1: !in_ty, %arg2: !in_ty, %cond: i1) -> !out_ty {
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi16>
  %plain = lwe.rlwe_encode %cst {encoding = #encoding, ring = #ring} : tensor<32xi16> -> !plain_ty

  %ret = scf.if %cond -> !out_ty {
    %0 = openfhe.mul %arg0, %arg1, %arg2 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
    %1 = openfhe.add %arg0, %0, %arg1 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
    %2 = openfhe.mul_plain %arg0, %0, %plain : (!ctxt_ty, !in_ty, !plain_ty) -> !in_ty
    %3 = lwe.reinterpret_underlying_type %1 : !in_ty to !out_ty
    scf.yield %3 : !out_ty
  } else {
    %4 = openfhe.mul_plain %arg0, %arg1, %plain : (!ctxt_ty, !in_ty, !plain_ty) -> !in_ty
    %5 = openfhe.mul %arg0, %4, %arg2 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
    %6 = openfhe.sub %arg0, %5, %4 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
    %7 = openfhe.mul_plain %arg0, %5, %plain : (!ctxt_ty, !in_ty, !plain_ty) -> !in_ty
    %8 = lwe.reinterpret_underlying_type %7 : !in_ty to !out_ty
    scf.yield %8 : !out_ty
    }
  return %ret : !out_ty
}

// CHECK: @complex_func
// CHECK: @complex_func__generate_crypto_context
// CHECK: mulDepth = 3

// CHECK: @complex_func__configure_crypto_context
// CHECK: openfhe.gen_mulkey
