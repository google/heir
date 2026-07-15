// RUN: heir-opt --mlir-print-local-scope --secret-to-ckks %s | FileCheck %s

// This checks a fix that mgmt.modreduce lowers to ckks.rescale without error.
// Before, the op verifier calculated the expected scaling factor using
// APInt::nearestLog2 which used a heuristic that caused the rescale operation
// to expect a different scaling factor result.

!efi = !secret.secret<tensor<8192xf32>>

module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [1073479681, 12451841], P = [22806529], logDefaultScale = 48, encryptionTechnique = extended>, scheme.ckks} {
  // CHECK: @test_rescale_scale_rounding
  // CHECK-SAME: !lwe.lwe_ciphertext<{{.*}}scaling_factor = 48{{.*}}>
  func.func @test_rescale_scale_rounding(%arg0: !efi {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 48>}) -> (!efi {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 24>}) {
    // CHECK: ckks.rescale %{{.*}} : !lwe.lwe_ciphertext<{{.*}}scaling_factor = 48{{.*}}> -> !lwe.lwe_ciphertext<{{.*}}scaling_factor = 24{{.*}}>
    // CHECK: return
    %0 = secret.generic(%arg0: !efi) {
    ^body(%input0: tensor<8192xf32>):
      %1 = mgmt.modreduce %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 24>} : tensor<8192xf32>
      secret.yield %1 : tensor<8192xf32>
    } -> (!efi {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 24>})
    return %0 : !efi
  }
}
