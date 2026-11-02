// RUN: heir-opt --lwe-to-cheddar-gpu %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>

!rns_L2 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099523555329 : i64>>
#ring_rns_L2 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix>
#chain_L2 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64>, current = 2>

!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #chain_L2>

// CHECK: func.func @test_rotate
// CHECK-SAME: cheddar_gpu.abi = {args = [{{.*}}], keys = [{galois = 5 : i64, rot_dist = 1 : i64, type = "rot"}, {galois = 25 : i64, rot_dist = 2 : i64, type = "rot"}], pts = [], results = [{{.*}}]}
// CHECK: cheddar_gpu.hrot %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, evk_bx(%{{.*}}), evk_ax(%{{.*}}) <level = 2, rot_dist = 1, galois = 5, in_form = eval, out_form = eval>
// CHECK: cheddar_gpu.hrot %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, evk_bx(%{{.*}}), evk_ax(%{{.*}}) <level = 2, rot_dist = 2, galois = 25, in_form = eval, out_form = eval>
// CHECK: cheddar_gpu.copy
// CHECK: cheddar_gpu.copy

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1125899908022273, 1099515691009, 1099523555329], P = [72057594038321153], logDefaultScale = 40>} {
  func.func @test_rotate(%a: !ct_L2) -> (!ct_L2, !ct_L2, !ct_L2) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %r1 = ckks.rotate %a, %c1 : index : !ct_L2
    %r2 = ckks.rotate %a, %c2 : index : !ct_L2
    %r0 = ckks.rotate %a, %c0 : index : !ct_L2
    return %r1, %r2, %r0 : !ct_L2, !ct_L2, !ct_L2
  }
}
