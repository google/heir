// RUN: heir-opt %s --canonicalize --validate-tensor-kernels --verify-diagnostics

// An index expression that normalization folds must remain supported.
func.func @folded_index(%x: tensor<4xf32> {secret.secret}) -> f32 {
  %one = arith.constant 1 : index
  %i = arith.addi %one, %one : index
  %y = tensor.extract %x[%i] : tensor<4xf32>
  return %y : f32
}
