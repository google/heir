// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus=#polynomial.int_polynomial<x**16384 + 1>>
#params = #lwe.rlwe_params<dimension=1, ring=#ring>
!cc = !openfhe.crypto_context
!ct = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type = i3>

func.func @test_binops(%cc : !cc, %input1 : !ct, %input2 : !ct) -> !ct {
  %add_res = openfhe.add %cc, %input1, %input2 : (!cc, !ct, !ct) -> !ct
  %sub_res = openfhe.sub %cc, %input1, %input2 : (!cc, !ct, !ct) -> !ct
  %mul_res = openfhe.mul %cc, %add_res, %sub_res : (!cc, !ct, !ct) -> !ct
  return %mul_res : !ct
}
