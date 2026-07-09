// RUN: heir-opt --mlir-to-bfv='ciphertext-degree=64 plaintext-modulus=65537' %s

!ct_s = !secret.secret<tensor<1x8xi32>>

func.func @repro(%a: !ct_s, %b: !ct_s) -> !secret.secret<tensor<2x8xi32>> {
  %0 = secret.generic(%a: !ct_s, %b: !ct_s) {
  ^body(%va: tensor<1x8xi32>, %vb: tensor<1x8xi32>):
    %empty = tensor.empty() : tensor<2x8xi32>
    %ins0 = tensor.insert_slice %va into %empty[0, 0] [1, 8] [1, 1]
        : tensor<1x8xi32> into tensor<2x8xi32>
    %ins1 = tensor.insert_slice %vb into %ins0[1, 0] [1, 8] [1, 1]
        : tensor<1x8xi32> into tensor<2x8xi32>
    secret.yield %ins1 : tensor<2x8xi32>
  } -> !secret.secret<tensor<2x8xi32>>
  return %0 : !secret.secret<tensor<2x8xi32>>
}
