// RUN: heir-opt --bgv-to-openfhe %s | FileCheck %s

#my_poly = #polynomial.polynomial<1 + x**1024>
#ring1 = #polynomial.ring<cmod=463187969, ideal=#my_poly>
#ring2 = #polynomial.ring<cmod=33538049, ideal=#my_poly>
#rings = #bgv.rings<#ring1, #ring2>
#rings2 = #bgv.rings<#ring1, #ring2, #ring2>
!ct = !bgv.ciphertext<rings=#rings, dim=2, level=1>
!ct_dim = !bgv.ciphertext<rings=#rings, dim=4, level=1>
!ct_level = !bgv.ciphertext<rings=#rings2, dim=2, level=2>

// CHECK: module
module {
  // CHECK: func.func @test_fn([[X:%.+]]: [[T:.*33538049.*]]) -> [[T]]
  func.func @test_fn(%x : !ct) -> !ct {
    // CHECK: return [[X]] : [[T]]
    return %x : !ct
  }

  // CHECK: func.func @test_ops([[C:%.+]]: [[S:.*crypto_context]], [[X:%.+]]: [[T:.*33538049.*]], [[Y:%.+]]: [[T]])
  func.func @test_ops(%x : !ct, %y : !ct) {
    // CHECK: %[[v1:.*]] = openfhe.negate [[C]], %[[x1:.*]] : ([[S]], [[T]]) -> [[T]]
    %negate = bgv.negate(%x) : !ct
    // CHECK: %[[v2:.*]] = openfhe.add [[C]], %[[x2:.*]], %[[y2:.*]]: ([[S]], [[T]], [[T]]) -> [[T]]
    %add = bgv.add(%x, %y) : !ct
    // CHECK: %[[v3:.*]] = openfhe.sub [[C]], %[[x3:.*]], %[[y3:.*]]: ([[S]], [[T]], [[T]]) -> [[T]]
    %sub = bgv.sub(%x, %y) : !ct
    // CHECK: %[[v4:.*]] = openfhe.mul_no_relin [[C]], %[[x4:.*]], %[[y4:.*]]: ([[S]], [[T]], [[T]]) -> [[T]]
    %mul = bgv.mul(%x, %y) : !ct -> !bgv.ciphertext<rings=#rings, dim=3, level=1>
    // CHECK: %[[c5:.*]] = arith.constant 4 : i64
    // CHECK: %[[v5:.*]] = openfhe.rot [[C]], %[[x5:.*]], %[[c5:.*]]: ([[S]], [[T]], i64) -> [[T]]
    %rot = bgv.rotate(%x) {offset = 4}: (!ct) -> !ct
    return
  }

  // CHECK: func.func @test_relin([[C]]: [[S]], [[X:%.+]]: [[T:.*33538049.*]])
  func.func @test_relin(%x : !ct_dim) {
    // CHECK: %[[v6:.*]] = openfhe.relin [[C]], %[[x6:.*]]: ([[S]], [[T]]) -> [[T]]
    %relin = bgv.relinearize(%x) {
      from_basis = array<i32: 0, 1, 2, 3>, to_basis = array<i32: 0, 1>
    }: (!ct_dim) -> !ct
    return
  }

  // CHECK: func.func @test_modswitch([[C]]: [[S]], [[X:%.+]]: [[T:.*33538049.*]])
  func.func @test_modswitch(%x : !ct_level) {
    // CHECK: %[[v7:.*]] = openfhe.mod_reduce [[C]], %[[x7:.*]]: ([[S]], [[T]]) -> [[T]]
    %mod_switch = bgv.modulus_switch(%x) {
      from_level = 2, to_level = 1
      }: (!ct_level) -> !bgv.ciphertext<rings=#rings2, dim=2, level=1>
    return
  }
}
