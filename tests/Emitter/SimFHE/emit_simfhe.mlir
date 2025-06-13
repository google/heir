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
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = i3>, plaintext_space = #plaintext_space>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>
!ct = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_D3 = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

// CHECK: import params
// CHECK-NEXT: import evaluator
// CHECK-NEXT: from perf_counter import PerfCounter
// CHECK-NEXT: import tabulate
// CHECK: def run_workload(


module {
  func.func @test_ops(%x : !ct, %y : !ct, %z : !pt) -> (!ct, !ct, !ct, !ct_D3, !ct, !ct, !ct, !ct) {
    // CHECK: def test_ops(
    // CHECK: stats = PerfCounter()
    %negate = ckks.negate %x  : !ct
    %add = ckks.add %x, %y  : (!ct, !ct) -> !ct
    %sub = ckks.sub %x, %y  : (!ct, !ct) -> !ct
    %mul = ckks.mul %x, %y  : (!ct, !ct) -> !ct_D3
    %relin = ckks.relinearize %mul  {
      from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>
    }: !ct_D3 -> !ct
    %mul_again = ckks.mul %relin, %x  : (!ct, !ct) -> !ct_D3
    %rot = ckks.rotate %x { offset = 4 } : !ct
    %add_plain = ckks.add_plain %x, %z : (!ct, !pt) -> !ct
    %sub_plain = ckks.sub_plain %x, %z : (!ct, !pt) -> !ct
    %mul_plain = ckks.mul_plain %x, %z : (!ct, !pt) -> !ct
    return %negate, %add, %sub, %mul_again, %rot, %add_plain, %sub_plain, %mul_plain : !ct, !ct, !ct, !ct_D3, !ct, !ct, !ct, !ct
  }
}




// CHECK: stats += evaluator.negate([[ARG0:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES0:[A-Za-z0-9_]+]] = [[ARG0]]
// CHECK: stats += evaluator.add([[ARG1:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES1:[A-Za-z0-9_]+]] = [[ARG1]]
// CHECK: stats += evaluator.subtract([[ARG2:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES2:[A-Za-z0-9_]+]] = [[ARG2]]
// CHECK: stats += evaluator.multiply([[ARG3:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES3:[A-Za-z0-9_]+]] = [[ARG3]]
// CHECK: stats += evaluator.rotate([[ARG4:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES4:[A-Za-z0-9_]+]] = [[ARG4]]
// CHECK: stats += evaluator.add_plain([[ARG5:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES5:[A-Za-z0-9_]+]] = [[ARG5]]
// CHECK: stats += evaluator.subtract_plain([[ARG6:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES6:[A-Za-z0-9_]+]] = [[ARG6]]
// CHECK: stats += evaluator.multiply_plain([[ARG7:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES7:[A-Za-z0-9_]+]] = [[ARG7]]
// CHECK: return stats

// CHECK: if __name__ == "__main__":
// CHECK: run_workload(test_ops)
