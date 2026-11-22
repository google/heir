// RUN: heir-opt --lwe-to-stablehlo %s | FileCheck %s
// RUN: heir-opt --lwe-to-stablehlo="r=2 batch=2" %s | FileCheck %s --check-prefix=CUSTOM

!Z2147473409 = !mod_arith.int<2147473409 : i64>
!Z1073563649 = !mod_arith.int<1073563649 : i64>
!rns_L1 = !rns.rns<!Z2147473409, !Z1073563649>

#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#chain_L1 = #lwe.modulus_chain<elements = <2147473409 : i64, 1073563649 : i64>, current = 1>
#chain_L0 = #lwe.modulus_chain<elements = <2147473409 : i64>, current = 0>

#ring_rns_16 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**16>>
#ring_single_16 = #polynomial.ring<coefficientType = !Z2147473409, polynomialModulus = <1 + x**16>>

#ct_space_rns = #lwe.ciphertext_space<ring = #ring_rns_16, encryption_type = ckks>
#ct_space_single = #lwe.ciphertext_space<ring = #ring_single_16, encryption_type = ckks>

!ct_rns = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = #ct_space_rns, key = #key, modulus_chain = #chain_L1>
!ct_single = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = #ct_space_single, key = #key, modulus_chain = #chain_L0>

// CHECK-NOT: lwe.
module {
  // CHECK: func.func @test_radd(%[[LHS:.*]]: tensor<1x2x4x4x2xui32>, %[[RHS:.*]]: tensor<1x2x4x4x2xui32>) -> tensor<1x2x4x4x2xui32>
  // CHECK: %[[LHS64:.*]] = stablehlo.convert %[[LHS]] : (tensor<1x2x4x4x2xui32>) -> tensor<1x2x4x4x2xui64>
  // CHECK: %[[RHS64:.*]] = stablehlo.convert %[[RHS]] : (tensor<1x2x4x4x2xui32>) -> tensor<1x2x4x4x2xui64>
  // CHECK: %[[SUM:.*]] = stablehlo.add %[[LHS64]], %[[RHS64]] : tensor<1x2x4x4x2xui64>
  // CHECK: %[[Q1D:.*]] = stablehlo.constant dense<[2147473409, 1073563649]> : tensor<2xui64>
  // CHECK: %[[Q5D:.*]] = stablehlo.broadcast_in_dim %[[Q1D]], dims = [4] : (tensor<2xui64>) -> tensor<1x2x4x4x2xui64>
  // CHECK: %[[GE:.*]] = stablehlo.compare GE, %[[SUM]], %[[Q5D]], UNSIGNED : (tensor<1x2x4x4x2xui64>, tensor<1x2x4x4x2xui64>) -> tensor<1x2x4x4x2xi1>
  // CHECK: %[[DIFF:.*]] = stablehlo.subtract %[[SUM]], %[[Q5D]] : tensor<1x2x4x4x2xui64>
  // CHECK: %[[SEL:.*]] = stablehlo.select %[[GE]], %[[DIFF]], %[[SUM]] : tensor<1x2x4x4x2xi1>, tensor<1x2x4x4x2xui64>
  // CHECK: %[[RES:.*]] = stablehlo.convert %[[SEL]] : (tensor<1x2x4x4x2xui64>) -> tensor<1x2x4x4x2xui32>
  // CHECK: return %[[RES]] : tensor<1x2x4x4x2xui32>

  // CUSTOM: func.func @test_radd(%{{.*}}: tensor<2x2x2x8x2xui32>, %{{.*}}: tensor<2x2x2x8x2xui32>) -> tensor<2x2x2x8x2xui32>
  func.func @test_radd(%ct0: !ct_rns, %ct1: !ct_rns) -> !ct_rns {
    %sum = lwe.radd %ct0, %ct1 : (!ct_rns, !ct_rns) -> !ct_rns
    return %sum : !ct_rns
  }

  // CHECK: func.func @test_radd_single_modulus(%{{.*}}: tensor<1x2x4x4x1xui32>, %{{.*}}: tensor<1x2x4x4x1xui32>) -> tensor<1x2x4x4x1xui32>
  // CHECK: stablehlo.constant dense<2147473409> : tensor<1xui64>
  func.func @test_radd_single_modulus(%ct0: !ct_single, %ct1: !ct_single) -> !ct_single {
    %sum = lwe.radd %ct0, %ct1 : (!ct_single, !ct_single) -> !ct_single
    return %sum : !ct_single
  }

  // CHECK: func.func @test_batched_tensor(%{{.*}}: tensor<4x2x4x4x2xui32>, %{{.*}}: tensor<4x2x4x4x2xui32>) -> tensor<4x2x4x4x2xui32>
  // CHECK: stablehlo.broadcast_in_dim %{{.*}}, dims = [4] : (tensor<2xui64>) -> tensor<4x2x4x4x2xui64>
  // CHECK: stablehlo.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x2x4x4x2xi1>, tensor<4x2x4x4x2xui64>
  // CUSTOM: func.func @test_batched_tensor(%{{.*}}: tensor<4x2x2x8x2xui32>, %{{.*}}: tensor<4x2x2x8x2xui32>) -> tensor<4x2x2x8x2xui32>
  func.func @test_batched_tensor(%ct0: tensor<4x!ct_rns>, %ct1: tensor<4x!ct_rns>) -> tensor<4x!ct_rns> {
    %sum = lwe.radd %ct0, %ct1 : (tensor<4x!ct_rns>, tensor<4x!ct_rns>) -> tensor<4x!ct_rns>
    return %sum : tensor<4x!ct_rns>
  }
}
