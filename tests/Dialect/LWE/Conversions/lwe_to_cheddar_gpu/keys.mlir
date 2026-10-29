// RUN: heir-opt --lwe-to-cheddar-gpu %s | FileCheck %s --check-prefix=FUSE-ON
// RUN: heir-opt --lwe-to-cheddar-gpu="fuse-relin-rescale=false" %s | FileCheck %s --check-prefix=FUSE-OFF

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>

!rns_L1 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>>
!rns_L2 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099523555329 : i64>>

#ring_rns_L1 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**65536>>
#ring_rns_L2 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**65536>>

#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1, encryption_type = mix>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix>
#ciphertext_space_L2_D3 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix, size = 3>

#chain_L1 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64>, current = 1>
#chain_L2 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64>, current = 2>

!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #chain_L1>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #chain_L2>
!ct_L2_D3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2_D3, key = #key, modulus_chain = #chain_L2>

// FUSE-ON-LABEL: func.func @test_relin_fuse
// FUSE-ON-SAME: cheddar_gpu.abi = {args = [{{.*}}], keys = [{type = "relin"}], pts = [], results = [{{.*}}]}
// FUSE-ON: cheddar_gpu.tensor
// FUSE-ON: cheddar_gpu.relinearize_rescale %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, evk_bx(%{{.*}}), evk_ax(%{{.*}}) <level = 2, in_form = eval, out_form = eval>
// FUSE-ON-NOT: cheddar_gpu.relinearize
// FUSE-ON-NOT: cheddar_gpu.rescale

// FUSE-OFF-LABEL: func.func @test_relin_fuse
// FUSE-OFF-SAME: cheddar_gpu.abi = {args = [{{.*}}], keys = [{type = "relin"}], pts = [], results = [{{.*}}]}
// FUSE-OFF: cheddar_gpu.tensor
// FUSE-OFF: cheddar_gpu.relinearize %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, evk_bx(%{{.*}}), evk_ax(%{{.*}}) <level = 2, in_form = eval, out_form = eval>
// FUSE-OFF: cheddar_gpu.rescale %{{.*}}, %{{.*}} <level = 2, in_form = eval, out_form = eval>
// FUSE-OFF: cheddar_gpu.rescale %{{.*}}, %{{.*}} <level = 2, in_form = eval, out_form = eval>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1125899908022273, 1099515691009, 1099523555329], P = [72057594038321153], logDefaultScale = 40>} {
  func.func @test_relin_fuse(%a: !ct_L2, %b: !ct_L2) -> !ct_L1 {
    %prod = ckks.mul %a, %b : (!ct_L2, !ct_L2) -> !ct_L2_D3
    %relin = ckks.relinearize %prod <from_basis = [0, 1, 2], to_basis = [0, 1]> : (!ct_L2_D3) -> !ct_L2
    %res = ckks.rescale %relin <to_ring = #ring_rns_L1> : !ct_L2 -> !ct_L1
    return %res : !ct_L1
  }
}
