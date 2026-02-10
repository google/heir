// RUN: heir-translate --emit-simfhe %s | FileCheck %s

!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>
!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_D3 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

//CHECK: import params
//CHECK: import evaluator
//CHECK: from perf_counter import PerfCounter
//CHECK: from experiment import run_mutiple, print_table, Target

module {
  func.func @test_ops(%x : !ct, %y : !ct, %z : !pt) -> (!ct, !ct, !ct, !ct_D3, !ct, !ct, !ct, !ct) {
    // CHECK: def test_ops([[CT:.*]], [[CT1:.*]], [[PT:.*]], scheme_params : params.SchemeParams):
    // CHECK: stats = PerfCounter()
    %negate = ckks.negate %x  : !ct
    // CHECK: stats += evaluator.multiply_plain([[CT]], scheme_params.arch_param)
    // CHECK:  [[CT2:ct[0-9]+]] = [[CT]]
    %add = ckks.add %x, %y  : (!ct, !ct) -> !ct
    // CHECK: stats += evaluator.add([[CT]], scheme_params.arch_param)
    // CHECK:  [[CT3:ct[0-9]+]] = [[CT]]
    %sub = ckks.sub %x, %y  : (!ct, !ct) -> !ct
    // CHECK: stats += evaluator.add([[CT]], scheme_params.arch_param)
    // CHECK:  [[CT4:ct[0-9]+]] = [[CT]]
    %mul = ckks.mul %x, %y  : (!ct, !ct) -> !ct_D3
    // CHECK: stats += evaluator.multiply([[CT]], scheme_params.arch_param)
    // CHECK:  [[CT5:ct[0-9]+]] = [[CT]]
    %relin = ckks.relinearize %mul  {
      from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>
    }: (!ct_D3) -> !ct
    // CHECK: stats += evaluator.key_switch([[CT5]], scheme_params.fresh_ctxt, scheme_params.arch_param)
    // CHECK:  [[CT6:ct[0-9]+]] = [[CT5]]
    %mul_again = ckks.mul %relin, %x  : (!ct, !ct) -> !ct_D3
    // CHECK: stats += evaluator.multiply([[CT6]], scheme_params.arch_param)
    // CHECK:  [[CT7:ct[0-9]+]] = [[CT6]]
    %rot = ckks.rotate %x { offset = 4 } : !ct
    // CHECK: stats += evaluator.rotate([[CT]], scheme_params.arch_param)
    // CHECK:  [[CT8:ct[0-9]+]] = [[CT]]
    %add_plain = ckks.add_plain %x, %z : (!ct, !pt) -> !ct
    // CHECK: stats += evaluator.add_plain([[CT]], scheme_params.arch_param)
    // CHECK:  [[CT9:ct[0-9]+]] = [[CT]]
    %sub_plain = ckks.sub_plain %x, %z : (!ct, !pt) -> !ct
    // CHECK: stats += evaluator.add_plain([[CT]], scheme_params.arch_param)
    // CHECK:  [[CT10:ct[0-9]+]] = [[CT]]
    %mul_plain = ckks.mul_plain %x, %z : (!ct, !pt) -> !ct
    // CHECK: stats += evaluator.multiply_plain([[CT]], scheme_params.arch_param)
    // CHECK:  [[CT11:ct[0-9]+]] = [[CT]]
    return %negate, %add, %sub, %mul_again, %rot, %add_plain, %sub_plain, %mul_plain : !ct, !ct, !ct, !ct_D3, !ct, !ct, !ct, !ct
  }
}


// CHECK: if __name__ == "__main__":
// CHECK:   targets = []
// CHECK:   for scheme_params in
// CHECK:     print(scheme_params)
// CHECK:     targets.append(Target("generated.test_ops",1, [scheme_params.fresh_ctxt,scheme_params.fresh_ctxt,params.PolyContext(scheme_params.fresh_ctxt.logq, scheme_params.fresh_ctxt.logN, scheme_params.fresh_ctxt.dnum,1), scheme_params]))
// CHECK:   headers, data = run_mutiple(targets)
// CHECK:   print_table(headers, data)
