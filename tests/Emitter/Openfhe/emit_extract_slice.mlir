// RUN: heir-translate %s --emit-openfhe-pke --split-input-file | FileCheck %s

// CHECK: std::vector<int32_t> emit_extract_slice_to_2d(std::vector<int32_t> v0) {
// CHECK:   std::vector<int32_t> v1(1024);
// CHECK:   for (int64_t v1_i0 = 0; v1_i0 < 1; ++v1_i0) {
// CHECK:     for (int64_t v1_i1 = 0; v1_i1 < 1024; ++v1_i1) {
// CHECK:       v1[v1_i1 + 1024 * (v1_i0)] = v0[0 + v1_i1 * 1 + 1024 * (0 + v1_i0 * 1)];
// CHECK:     }
// CHECK:   }
// CHECK:   return v1;
// CHECK: }

module attributes {scheme.bgv} {
  func.func @emit_extract_slice_to_2d(%arg0: tensor<10x1024xi32>) -> tensor<1x1024xi32> {
    %0 = tensor.extract_slice %arg0[0, 0] [1, 1024] [1, 1] : tensor<10x1024xi32> to tensor<1x1024xi32>
    func.return %0 : tensor<1x1024xi32>
  }
}

// -----

// CHECK: emit_extract_slice_to_1d
// CHECK-COUNT-2: for (

// Should be no different from above, despite the result type in MLIR being rank-reduced.

// CHECK:       v1[v1_i1 + 1024 * (v1_i0)] = v0[0 + v1_i1 * 1 + 1024 * (0 + v1_i0 * 1)];

module attributes {scheme.bgv} {
  func.func @emit_extract_slice_to_1d(%arg0: tensor<10x1024xi32>) -> tensor<1024xi32> {
    %0 = tensor.extract_slice %arg0[0, 0] [1, 1024] [1, 1] : tensor<10x1024xi32> to tensor<1024xi32>
    func.return %0 : tensor<1024xi32>
  }
}

// -----

// CHECK: emit_extract_slice_with_nontrivial_params
// CHECK: std::vector<int32_t> v1(512);
// CHECK: for (int64_t v1_i0 = 0; v1_i0 < 2; ++v1_i0) {
// CHECK:   for (int64_t v1_i1 = 0; v1_i1 < 256; ++v1_i1) {
// CHECK:     v1[v1_i1 + 256 * (v1_i0)] = v0[3 + v1_i1 * 3 + 1024 * (2 + v1_i0 * 2)];

module attributes {scheme.bgv} {
  func.func @emit_extract_slice_with_nontrivial_params(%arg0: tensor<10x1024xi32>) -> tensor<2x256xi32> {
    %0 = tensor.extract_slice %arg0[2, 3] [2, 256] [2, 3] : tensor<10x1024xi32> to tensor<2x256xi32>
    func.return %0 : tensor<2x256xi32>
  }
}
