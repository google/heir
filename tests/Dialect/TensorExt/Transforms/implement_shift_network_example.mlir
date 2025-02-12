// RUN: heir-opt --implement-shift-network=ciphertext-size=16 %s | FileCheck %s

#map = dense<[13, 8, 4, 0, 11, 7, 14, 5, 15, 3, 12, 6, 10, 2, 9, 1]> : tensor<16xi64>
// CHECK-LABEL: @figure3
func.func @figure3(%0: tensor<16xi32>) -> tensor<16xi32> {
  %1 = tensor_ext.permute %0 {permutation = #map} : tensor<16xi32>
  return %1 : tensor<16xi32>
}
