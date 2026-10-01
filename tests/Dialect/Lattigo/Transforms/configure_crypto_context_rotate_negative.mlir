// RUN: heir-opt --lattigo-configure-crypto-context=entry-function=rotate_negative %s | FileCheck %s

!evaluator = !lattigo.ckks.evaluator
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @rotate_negative(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %shift = arith.constant -512 : index
    %res = lattigo.ckks.rotate_new %evaluator, %ct, %shift : (!evaluator, !ct, index) -> !ct
    return %res : !ct
  }
}

// A rotation by -512 with logN = 13 requires the galois element
// 5^(-512 mod 2N) mod 2N = 5^15872 mod 16384 = 2049, not 1.
// CHECK: @rotate_negative
// CHECK: @rotate_negative__configure
// CHECK: lattigo.rlwe.gen_galois_key
// CHECK-SAME: galoisElement = 2049
