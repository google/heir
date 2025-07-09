// Single (mini) sbox table lookup for AES encryption.

#map = affine_map<(d0) -> (d0)>
module {
  func.func @sub_bytes(%arg0: i8 {secret.secret}) -> i8 {
    %cst = arith.constant dense<"0x637C777BF26B6FC5"> : tensor<8xi8>
    %2 = arith.index_cast %arg0 : i8 to index
    %extracted = tensor.extract %cst[%2] : tensor<8xi8>
    return %extracted : i8
  }
}
