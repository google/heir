// RUN: heir-opt --lwe-to-cheddar --cheddar-bufferize --canonicalize %s | FileCheck %s

#enc = #lwe.inverse_canonical_encoding<scaling_factor = 1099511627776>
#key = #lwe.key<>
#chain = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64>, current = 1>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns = !rns.rns<!mod_arith.int<36028797019488257 : i64>, !mod_arith.int<1099512938497 : i64>>
#ring_rns = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**65536>>
#ct_space = #lwe.ciphertext_space<ring = #ring_rns, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64, encoding = #enc>, ciphertext_space = #ct_space, key = #key, modulus_chain = #chain>

module attributes {
  backend.cheddar,
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<
    logN = 16,
    Q = [36028797019488257, 1099512938497],
    P = [2305843009211596801],
    logDefaultScale = 40
  >
} {
  // One-Shot Bufferize chooses storage for the ciphertext result. Promoting
  // the returned allocation to an out-param makes it caller-owned.
  // CHECK: func.func private @leaf(
  // CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<!ciphertext> {bufferize.result}
  // CHECK-NOT: memref.alloc
  // CHECK: cheddar.add {{.*}}, %[[OUT]]
  // CHECK-NOT: memref.copy
  // CHECK: return
  func.func private @leaf(%lhs: !ct, %rhs: !ct) -> !ct {
    %sum = lwe.radd %lhs, %rhs : (!ct, !ct) -> !ct
    return %sum : !ct
  }

  // An immediately returned call forwards its caller-provided destination to
  // the callee directly. No temporary tensor result remains at the call.
  // CHECK: func.func @forward(
  // CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<!ciphertext> {bufferize.result}
  // CHECK: %[[TMP:.*]] = memref.alloc() : memref<!ciphertext>
  // CHECK: call @leaf({{.*}}, %[[TMP]])
  // CHECK: memref.copy %[[TMP]], %[[OUT]]
  // CHECK: return
  func.func @forward(%lhs: !ct, %rhs: !ct) -> !ct {
    %result = func.call @leaf(%lhs, %rhs) : (!ct, !ct) -> !ct
    return %result : !ct
  }
}
