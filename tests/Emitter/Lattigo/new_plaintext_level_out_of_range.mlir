// RUN: not heir-translate %s --emit-lattigo --split-input-file 2>&1 | FileCheck %s

!pt = !lattigo.rlwe.plaintext
!params = !lattigo.ckks.parameter
#paramsLiteral = #lattigo.ckks.parameters_literal<logN = 14, logQ = [55, 45, 45], logP = [61], logDefaultScale = 45>

module attributes {scheme.ckks} {
  func.func @ckks_level_past_chain() {
    %params = lattigo.ckks.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !params
    // CHECK: level 5 is past the top of the modulus chain
    // CHECK-SAME: 3 moduli
    // CHECK-SAME: maximum level of 2
    %pt = lattigo.ckks.new_plaintext %params {level = 5 : i64} : (!params) -> !pt
    return
  }
}

// -----

!pt = !lattigo.rlwe.plaintext
!params = !lattigo.bgv.parameter
#paramsLiteral = #lattigo.bgv.parameters_literal<logN = 14, Q = [36028797019389953, 35184372121601], P = [36028797019488257], plaintextModulus = 65537>

module attributes {scheme.bgv} {
  func.func @bgv_level_past_chain() {
    %params = lattigo.bgv.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !params
    // CHECK: level 4 is past the top of the modulus chain
    // CHECK-SAME: 2 moduli
    // CHECK-SAME: maximum level of 1
    %pt = lattigo.bgv.new_plaintext %params {level = 4 : i64} : (!params) -> !pt
    return
  }
}
