// RUN: heir-opt %s --split-preprocessing | FileCheck %s

// A loop carries a ciphertext iter_arg (the running accumulator) and also
// encodes a plaintext per iteration. The ciphertext value is not part of the
// plaintext slice, so when the loop is cloned into the preprocessing function
// its iter_arg becomes dead. remove-dead-values only poisons (it cannot
// structurally strip) a dead affine.for iter_arg, so split-preprocessing must
// remove it itself -- otherwise a ub.poison-typed loop-carried value survives
// and later backend lowering can neither convert nor emit it.

!Z36028797017456641_i64 = !mod_arith.int<36028797017456641 : i64>
!Z35184371138561_i64 = !mod_arith.int<35184371138561 : i64>
!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L2 = !rns.rns<!Z36028797017456641_i64, !Z35184371138561_i64, !Z35184372121601_i64>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L2_1_x1024 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x1024, encryption_type = mix>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #lwe.modulus_chain<elements = <36028797017456641 : i64, 35184371138561 : i64, 35184372121601 : i64>, current = 2>>

// The preprocessing loop must be a pure store loop: no iter_args, no ub.poison.
// CHECK:       func.func @f__preprocessing() -> !preprocessing.storage<!pt>
// CHECK-NOT:     ub.poison
// CHECK:         affine.for %[[I:.*]] = 0 to 4 {
// CHECK-NOT:     iter_args
// CHECK:           %[[PT:.*]] = lwe.rlwe_encode
// CHECK:           preprocessing.store %[[PT]], %{{.*}}[%[[I]]] site 0<!pt> : !pt, <!pt>
// CHECK:         return

// A second iter_args entry carries a plaintext accumulator. The encode op needs
// it, so it must survive in the preprocessing function while the ciphertext
// iter_arg is stripped. The kept entry moves from index 1 to index 0, so this
// covers the index remapping that a fully dead iter_args list does not reach.
// The rebuilt loop must also keep the discardable attributes of the original.
// CHECK:       func.func @two_iter_args__preprocessing() -> !preprocessing.storage<!pt>
// CHECK-NOT:     ub.poison
// CHECK:         %[[CST:.*]] = arith.constant dense
// CHECK:         affine.for %[[I:.*]] = 0 to 4 iter_args(%[[SCALE:.*]] = %[[CST]]) -> (tensor<1024xf32>) {
// CHECK:           %[[NEXT:.*]] = arith.addf %[[SCALE]], %[[CST]]
// CHECK:           %[[PT:.*]] = lwe.rlwe_encode %[[NEXT]]
// CHECK:           preprocessing.store %[[PT]], %{{.*}}[%[[I]]]
// CHECK:           affine.yield %[[NEXT]] : tensor<1024xf32>
// CHECK:         } {test.keep_me}
// CHECK:         return

// The preprocessed function loses the plaintext accumulator, because the encode
// op that used it moved to the preprocessing function. That dead iter_arg must
// be stripped too, for the same reason: remove-dead-values poisons it, and a
// ub.poison loop-carried value reaches backend lowering.
// CHECK:       func.func @two_iter_args__preprocessed
// CHECK-NOT:     ub.poison
// CHECK:         affine.for %{{.*}} = 0 to 4 iter_args(%[[SUM:.*]] = %{{.*}}) -> (tensor<1x!ct_L2>) {
// CHECK-NOT:       ub.poison
// CHECK:           %[[LOADED:.*]] = preprocessing.load
// CHECK:           %[[FROM:.*]] = tensor.from_elements %[[LOADED]]
// CHECK:           %[[ADD:.*]] = ckks.add_plain %[[SUM]], %[[FROM]]
// CHECK:           affine.yield %[[ADD]] : tensor<1x!ct_L2>
// CHECK:         return

// Two nested loops carry the same ciphertext accumulator. Both lose their
// iter_arg. The inner loop is rewritten first, and only that removal makes the
// outer iter_arg dead, so this covers the cascade and the move of an
// already-rewritten inner loop into its new parent.
// CHECK:       func.func @nested_loops__preprocessing() -> !preprocessing.storage<!pt>
// CHECK-NOT:     ub.poison
// CHECK:         affine.for %[[I:.*]] = 0 to 2 {
// CHECK-NOT:       iter_args
// CHECK:           affine.for %[[J:.*]] = 0 to 3 {
// CHECK:             %[[PT:.*]] = lwe.rlwe_encode
// CHECK:             preprocessing.store %[[PT]], %{{.*}}[%[[I]], %[[J]]]
// CHECK:         return

module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184371138561, 35184372121601], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 45>, scheme.ckks} {
  func.func @f(%arg0: tensor<1x!ct_L2>) -> tensor<1x!ct_L2> {
    %cst = arith.constant dense<1.0> : tensor<1024xf32>
    %0 = affine.for %i = 0 to 4 iter_args(%sum = %arg0) -> (tensor<1x!ct_L2>) {
      %pt = lwe.rlwe_encode %cst {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
      %from = tensor.from_elements %pt : tensor<1x!pt>
      %1 = ckks.add_plain %sum, %from : (tensor<1x!ct_L2>, tensor<1x!pt>) -> tensor<1x!ct_L2>
      affine.yield %1 : tensor<1x!ct_L2>
    }
    return %0 : tensor<1x!ct_L2>
  }

  func.func @two_iter_args(%arg0: tensor<1x!ct_L2>) -> tensor<1x!ct_L2> {
    %cst = arith.constant dense<1.0> : tensor<1024xf32>
    %sum, %scale = affine.for %i = 0 to 4 iter_args(%sum_iter = %arg0, %scale_iter = %cst) -> (tensor<1x!ct_L2>, tensor<1024xf32>) {
      %next = arith.addf %scale_iter, %cst : tensor<1024xf32>
      %pt = lwe.rlwe_encode %next {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
      %from = tensor.from_elements %pt : tensor<1x!pt>
      %0 = ckks.add_plain %sum_iter, %from : (tensor<1x!ct_L2>, tensor<1x!pt>) -> tensor<1x!ct_L2>
      affine.yield %0, %next : tensor<1x!ct_L2>, tensor<1024xf32>
    } {test.keep_me}
    return %sum : tensor<1x!ct_L2>
  }

  func.func @nested_loops(%arg0: tensor<1x!ct_L2>) -> tensor<1x!ct_L2> {
    %cst = arith.constant dense<1.0> : tensor<1024xf32>
    %0 = affine.for %i = 0 to 2 iter_args(%outer = %arg0) -> (tensor<1x!ct_L2>) {
      %1 = affine.for %j = 0 to 3 iter_args(%inner = %outer) -> (tensor<1x!ct_L2>) {
        %pt = lwe.rlwe_encode %cst {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
        %from = tensor.from_elements %pt : tensor<1x!pt>
        %2 = ckks.add_plain %inner, %from : (tensor<1x!ct_L2>, tensor<1x!pt>) -> tensor<1x!ct_L2>
        affine.yield %2 : tensor<1x!ct_L2>
      }
      affine.yield %1 : tensor<1x!ct_L2>
    }
    return %0 : tensor<1x!ct_L2>
  }
}
