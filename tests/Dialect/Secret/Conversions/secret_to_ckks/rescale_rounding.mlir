// RUN: heir-opt --mlir-print-local-scope --split-input-file --secret-to-ckks %s | FileCheck %s

// This checks a fix that mgmt.modreduce lowers to ckks.rescale without error.
// Before, the op verifier calculated the expected scaling factor using
// APInt::nearestLog2 which used a heuristic that caused the rescale operation
// to expect a different scaling factor result.

!efi = !secret.secret<tensor<8192xf32>>

module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [1073479681, 12451841], P = [22806529], logDefaultScale = 48, encryptionTechnique = extended>, scheme.ckks} {
  // CHECK: @test_rescale_scale_rounding
  // CHECK-SAME: !lwe.lwe_ciphertext<{{.*}}scaling_factor = 48{{.*}}!rns.rns<!mod_arith.int<1073479681 : i32>, !mod_arith.int<12451841 : i32>>{{.*}}>
  func.func @test_rescale_scale_rounding(%arg0: !efi {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 48>}) -> (!efi {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 24>}) {
    // CHECK: ckks.rescale %{{.*}} : !lwe.lwe_ciphertext<{{.*}}scaling_factor = 48{{.*}}> -> !lwe.lwe_ciphertext<{{.*}}scaling_factor = 24{{.*}}!rns.rns<!mod_arith.int<1073479681 : i32>>{{.*}}>
    // CHECK: return
    %0 = secret.generic(%arg0: !efi) {
    ^body(%input0: tensor<8192xf32>):
      %1 = mgmt.modreduce %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 24>} : tensor<8192xf32>
      secret.yield %1 : tensor<8192xf32>
    } -> (!efi {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 24>})
    return %0 : !efi
  }
}

// -----

!efi = !secret.secret<tensor<8192xf32>>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [1073692673, 1073643521, 268468225, 268369921], P = [1073709057, 1073741825], logDefaultScale = 56, moduliPerLevel = [2, 2]>, scheme.ckks} {
  // CHECK: @test_multi_modulus_level_rescale
  // CHECK-SAME: %[[arg0:[^:]+]]: !lwe.lwe_ciphertext<{{.*}}scaling_factor = 112{{.*}}!rns.rns<!mod_arith.int<1073692673 : i32>, !mod_arith.int<1073643521 : i32>, !mod_arith.int<268468225 : i32>, !mod_arith.int<268369921 : i32>>{{.*}}current = 3>>, %[[arg1:[^:]+]]: !lwe.lwe_ciphertext<{{.*}}scaling_factor = 56{{.*}}!rns.rns<!mod_arith.int<1073692673 : i32>, !mod_arith.int<1073643521 : i32>, !mod_arith.int<268468225 : i32>, !mod_arith.int<268369921 : i32>>{{.*}}current = 3>>
  func.func @test_multi_modulus_level_rescale(
      %arg0: !efi {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 112>},
      %arg1: !efi {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 56>}) -> (!efi {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 56>}) {
    // CHECK: ckks.rescale %[[arg0]] <to_ring = <coefficientType = !rns.rns<!mod_arith.int<1073692673 : i32>, !mod_arith.int<1073643521 : i32>>, polynomialModulus = <1 + x**8192>>>
    // CHECK-SAME: -> !lwe.lwe_ciphertext<{{.*}}scaling_factor = 56{{.*}}!rns.rns<!mod_arith.int<1073692673 : i32>, !mod_arith.int<1073643521 : i32>>{{.*}}current = 1>>
    %0 = secret.generic(%arg0: !efi) {
    ^body(%input0: tensor<8192xf32>):
      %1 = mgmt.modreduce %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 56>} : tensor<8192xf32>
      secret.yield %1 : tensor<8192xf32>
    } -> (!efi {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 56>})
    // CHECK: ckks.level_reduce %[[arg1]] <levelToDrop = 2>
    // CHECK-SAME: -> !lwe.lwe_ciphertext<{{.*}}scaling_factor = 56{{.*}}!rns.rns<!mod_arith.int<1073692673 : i32>, !mod_arith.int<1073643521 : i32>>{{.*}}current = 1>>
    %1 = secret.generic(%arg1: !efi) {
    ^body(%input1: tensor<8192xf32>):
      %2 = mgmt.level_reduce %input1 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 56>} : tensor<8192xf32>
      secret.yield %2 : tensor<8192xf32>
    } -> (!efi {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 56>})
    // CHECK: ckks.add
    %2 = secret.generic(%0: !efi, %1: !efi) {
    ^body(%lhs: tensor<8192xf32>, %rhs: tensor<8192xf32>):
      %3 = arith.addf %lhs, %rhs : tensor<8192xf32>
      secret.yield %3 : tensor<8192xf32>
    } -> (!efi {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 56>})
    // CHECK: return
    return %2 : !efi
  }
}
