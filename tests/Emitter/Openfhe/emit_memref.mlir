// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s --check-prefix=CHECK-SRC
// RUN: heir-translate %s --emit-openfhe-pke-header | FileCheck %s --check-prefix=CHECK-HDR

!cc = !openfhe.crypto_context
!pt = !openfhe.plaintext
!pk = !openfhe.public_key
!ct = !openfhe.ciphertext

module attributes {scheme.bgv} {
  func.func @test_memref(%arg0: memref<2x3xf32>, %v: f32) -> f32 {
    // CHECK-SRC: float test_memref(const std::vector<float>& [[arg0:[^ ]*]], float [[v:[^ ]*]])
    // CHECK-HDR: float test_memref(const std::vector<float>& [[arg0:[^ ]*]], float [[v:[^ ]*]]);
    // CHECK-SRC: std::vector<float> [[v0:[^ ]*]](6);
    %0 = memref.alloc() : memref<2x3xf32>

    // CHECK-SRC: [[v0]][2 + 3 * (1)] = [[v]];
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    memref.store %v, %0[%c1, %c2] : memref<2x3xf32>

    // CHECK-SRC: float [[v1:[^ ]*]] = [[arg0]][2 + 3 * (1)];
    %1 = memref.load %arg0[%c1, %c2] : memref<2x3xf32>

    return %1 : f32
  }

  func.func @test_mutated_memref(%arg0: memref<2x3xf32>, %v: f32) {
    // CHECK-SRC: void test_mutated_memref(std::vector<float>& [[arg0:[^ ]*]], float [[v:[^ ]*]])
    // CHECK-HDR: void test_mutated_memref(std::vector<float>& [[arg0:[^ ]*]], float [[v:[^ ]*]]);
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    memref.store %v, %arg0[%c1, %c2] : memref<2x3xf32>
    return
  }
}
