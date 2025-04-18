module attributes {tf_saved_model.semantics} {
  func.func @fn_under_test(%arg0: !tfhe_rust.server_key, %arg1: memref<2x3x!tfhe_rust.eui32>) -> memref<1x1x!tfhe_rust.eui32> {
    %c429_i32 = arith.constant 429 : i32
    %0 = tfhe_rust.create_trivial %arg0, %c429_i32 : (!tfhe_rust.server_key, i32) -> !tfhe_rust.eui32
    %c33_i32 = arith.constant 33 : i32
    %c0 = arith.constant 0 : index
    %1 = tfhe_rust.create_trivial %arg0, %c33_i32 : (!tfhe_rust.server_key, i32) -> !tfhe_rust.eui32
    %2 = affine.load %arg1[0, 0] : memref<2x3x!tfhe_rust.eui32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x!tfhe_rust.eui32>
    %3 = tfhe_rust.mul %arg0, %2, %1 : (!tfhe_rust.server_key, !tfhe_rust.eui32, !tfhe_rust.eui32) -> !tfhe_rust.eui32
    %4 = tfhe_rust.add %arg0, %3, %0 : (!tfhe_rust.server_key, !tfhe_rust.eui32, !tfhe_rust.eui32) -> !tfhe_rust.eui32
    affine.store %4, %alloc[0, 0] : memref<1x1x!tfhe_rust.eui32>
    return %alloc : memref<1x1x!tfhe_rust.eui32>
  }
}
