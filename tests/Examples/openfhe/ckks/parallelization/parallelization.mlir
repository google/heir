!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!params = !openfhe.cc_params
!pk = !openfhe.public_key
!pt = !openfhe.plaintext
!sk = !openfhe.private_key
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 1023 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1024xi16>, layout = #layout>
module attributes {backend.openfhe, scheme.ckks} {
  func.func @rotations(%cc: !cc, %arg0: tensor<1x!ct> {tensor_ext.original_type = #original_type}) -> tensor<25x!ct> {
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg0[%c0] : tensor<1x!ct>
    %ct1 = openfhe.rot %cc, %extracted {static_shift = 1 : index} : (!cc, !ct) -> !ct
    %ct2 = openfhe.rot %cc, %extracted {static_shift = 2 : index} : (!cc, !ct) -> !ct
    %ct3 = openfhe.rot %cc, %extracted {static_shift = 3 : index} : (!cc, !ct) -> !ct
    %ct4 = openfhe.rot %cc, %extracted {static_shift = 4 : index} : (!cc, !ct) -> !ct
    %ct5 = openfhe.rot %cc, %extracted {static_shift = 5 : index} : (!cc, !ct) -> !ct
    %ct6 = openfhe.rot %cc, %extracted {static_shift = 6 : index} : (!cc, !ct) -> !ct
    %ct7 = openfhe.rot %cc, %extracted {static_shift = 7 : index} : (!cc, !ct) -> !ct
    %ct8 = openfhe.rot %cc, %extracted {static_shift = 8 : index} : (!cc, !ct) -> !ct
    %ct9 = openfhe.rot %cc, %extracted {static_shift = 9 : index} : (!cc, !ct) -> !ct
    %ct10 = openfhe.rot %cc, %extracted {static_shift = 10 : index} : (!cc, !ct) -> !ct
    %ct11 = openfhe.rot %cc, %extracted {static_shift = 11 : index} : (!cc, !ct) -> !ct
    %ct12 = openfhe.rot %cc, %extracted {static_shift = 12 : index} : (!cc, !ct) -> !ct
    %ct13 = openfhe.rot %cc, %extracted {static_shift = 13 : index} : (!cc, !ct) -> !ct
    %ct14 = openfhe.rot %cc, %extracted {static_shift = 14 : index} : (!cc, !ct) -> !ct
    %ct15 = openfhe.rot %cc, %extracted {static_shift = 15 : index} : (!cc, !ct) -> !ct
    %ct16 = openfhe.rot %cc, %extracted {static_shift = 16 : index} : (!cc, !ct) -> !ct
    %ct17 = openfhe.rot %cc, %extracted {static_shift = 17 : index} : (!cc, !ct) -> !ct
    %ct18 = openfhe.rot %cc, %extracted {static_shift = 18 : index} : (!cc, !ct) -> !ct
    %ct19 = openfhe.rot %cc, %extracted {static_shift = 19 : index} : (!cc, !ct) -> !ct
    %ct20 = openfhe.rot %cc, %extracted {static_shift = 20 : index} : (!cc, !ct) -> !ct
    %ct21 = openfhe.rot %cc, %extracted {static_shift = 21 : index} : (!cc, !ct) -> !ct
    %ct22 = openfhe.rot %cc, %extracted {static_shift = 22 : index} : (!cc, !ct) -> !ct
    %ct23 = openfhe.rot %cc, %extracted {static_shift = 23 : index} : (!cc, !ct) -> !ct
    %ct24 = openfhe.rot %cc, %extracted {static_shift = 24 : index} : (!cc, !ct) -> !ct
    %ct25 = openfhe.rot %cc, %extracted {static_shift = 25 : index} : (!cc, !ct) -> !ct
    %res = tensor.from_elements %ct1, %ct2, %ct3, %ct4, %ct5, %ct6, %ct7, %ct8, %ct9, %ct10, %ct11, %ct12, %ct13, %ct14, %ct15, %ct16, %ct17, %ct18, %ct19, %ct20, %ct21, %ct22, %ct23, %ct24, %ct25 : tensor<25x!ct>
    return %res : tensor<25x!ct>
  }
  func.func @rotations__encrypt__arg0(%cc: !cc, %arg0: tensor<1024xi16>, %pk: !pk) -> tensor<1x!ct> attributes {client.enc_func = {func_name = "rotations", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0> : tensor<1x1024xi16>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = scf.for %arg1 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<1x1024xi16>)  : i32 {
      %2 = arith.index_cast %arg1 : i32 to index
      %extracted = tensor.extract %arg0[%2] : tensor<1024xi16>
      %inserted = tensor.insert %extracted into %arg2[%c0, %2] : tensor<1x1024xi16>
      scf.yield %inserted : tensor<1x1024xi16>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 1024] [1, 1] : tensor<1x1024xi16> to tensor<1024xi16>
    %1 = arith.extsi %extracted_slice : tensor<1024xi16> to tensor<1024xi64>
    %pt = openfhe.make_ckks_packed_plaintext %cc, %1 : (!cc, tensor<1024xi64>) -> !pt
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %from_elements = tensor.from_elements %ct : tensor<1x!ct>
    return %from_elements : tensor<1x!ct>
  }
  func.func @rotations__decrypt__result0(%cc: !cc, %arg0: tensor<1x!ct>, %sk: !sk) -> tensor<1024xi16> attributes {client.dec_func = {func_name = "rotations", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<1024xi16>
    %extracted = tensor.extract %arg0[%c0] : tensor<1x!ct>
    %pt = openfhe.decrypt %cc, %extracted, %sk : (!cc, !ct, !sk) -> !pt
    %0 = openfhe.decode_ckks %pt : !pt -> tensor<1x1024xi16>
    %1 = scf.for %arg1 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<1024xi16>)  : i32 {
      %2 = arith.index_cast %arg1 : i32 to index
      %extracted_0 = tensor.extract %0[%c0, %2] : tensor<1x1024xi16>
      %inserted = tensor.insert %extracted_0 into %arg2[%2] : tensor<1024xi16>
      scf.yield %inserted : tensor<1024xi16>
    }
    return %1 : tensor<1024xi16>
  }
  func.func @rotations__generate_crypto_context() -> !cc {
    %params = openfhe.gen_params  {mulDepth = 0 : i64, plainMod = 0 : i64} : () -> !params
    %cc = openfhe.gen_context %params {supportFHE = false} : (!params) -> !cc
    return %cc : !cc
  }
  func.func @rotations__configure_crypto_context(%cc: !cc, %sk: !sk) -> !cc {
    openfhe.gen_rotkey %cc, %sk {indices = array<i64: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25>} : (!cc, !sk) -> ()
    return %cc : !cc
  }
}
