// RUN: heir-opt --bgv-to-openfhe %s | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.polynomial<1 + x**1024>
// cmod is 64153 * 2521
#ring1 = #polynomial.ring<cmod=161729713, ideal=#my_poly>
#ring2 = #polynomial.ring<cmod=2521, ideal=#my_poly>

#params1 = #lwe.rlwe_params<dimension=2, ring=#ring1>
#params2 = #lwe.rlwe_params<dimension=4, ring=#ring1>
#params3 = #lwe.rlwe_params<dimension=2, ring=#ring2>
#params4 = #lwe.rlwe_params<dimension=3, ring=#ring1>

!ct = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1>
!ct_dim = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params2>
!ct_level = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params3>
!ct_level3 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params4>

// CHECK: module
module {
  // CHECK: func.func @test_fn([[X:%.+]]: [[T:.*161729713.*]]) -> [[T]]
  func.func @test_fn(%x : !ct) -> !ct {
    // CHECK: return [[X]] : [[T]]
    return %x : !ct
  }

  // CHECK: func.func @test_ops([[C:%.+]]: [[S:.*crypto_context]], [[X:%.+]]: [[T:.*161729713.*]], [[Y:%.+]]: [[T]])
  func.func @test_ops(%x : !ct, %y : !ct) {
    // CHECK: %[[v1:.*]] = openfhe.negate [[C]], %[[x1:.*]] : ([[S]], [[T]]) -> [[T]]
    %negate = bgv.negate(%x) : !ct
    // CHECK: %[[v2:.*]] = openfhe.add [[C]], %[[x2:.*]], %[[y2:.*]]: ([[S]], [[T]], [[T]]) -> [[T]]
    %add = bgv.add(%x, %y) : !ct
    // CHECK: %[[v3:.*]] = openfhe.sub [[C]], %[[x3:.*]], %[[y3:.*]]: ([[S]], [[T]], [[T]]) -> [[T]]
    %sub = bgv.sub(%x, %y) : !ct
    // CHECK: %[[v4:.*]] = openfhe.mul_no_relin [[C]], %[[x4:.*]], %[[y4:.*]]: ([[S]], [[T]], [[T]]) -> [[T]]
    %mul = bgv.mul(%x, %y) : !ct -> !ct_level3
    // CHECK: %[[c5:.*]] = arith.constant 4 : i64
    // CHECK: %[[v5:.*]] = openfhe.rot [[C]], %[[x5:.*]], %[[c5:.*]]: ([[S]], [[T]], i64) -> [[T]]
    %rot = bgv.rotate(%x) {offset = 4}: (!ct) -> !ct
    return
  }

  // CHECK: func.func @test_relin([[C]]: [[S]], [[X:%.+]]: [[T:.*161729713.*]])
  func.func @test_relin(%x : !ct_dim) {
    // CHECK: %[[v6:.*]] = openfhe.relin [[C]], %[[x6:.*]]: ([[S]], [[T]]) -> [[T]]
    %relin = bgv.relinearize(%x) {
      from_basis = array<i32: 0, 1, 2, 3>, to_basis = array<i32: 0, 1>
    }: (!ct_dim) -> !ct
    return
  }

  // CHECK: func.func @test_modswitch([[C]]: [[S]], [[X:%.+]]: [[T:.*161729713.*]]) -> [[T1:.*2521.*]] {
  func.func @test_modswitch(%x : !ct) -> !ct_level {
    // CHECK: %[[v7:.*]] = openfhe.mod_reduce [[C]], %[[x7:.*]] : ([[S]], [[T]]) -> [[T1]]
    %mod_switch = bgv.modulus_switch(%x) { to_ring=#ring2 }: (!ct) -> !ct_level
    return %mod_switch : !ct_level
  }
}
