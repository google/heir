// RUN: heir-opt --tensor-linalg-to-affine-loops --arith-to-cggi --verify-diagnostics -split-input-file %s

module {
  func.func @add_tensor(%arg0: tensor<4xi16> {secret.secret}, %arg1: tensor<4xi16> {secret.secret}) -> tensor<4xi16> {
    // expected-error@+1 {{--arith-to-cggi does not support arith operations on vectors or tensors. Lower them to scalars first.}}
    %0 = arith.addi %arg0, %arg1 : tensor<4xi16>
    return %0 : tensor<4xi16>
  }
}

// -----

module {
  func.func @rotate_tensor(%arg0: tensor<4xi16> {secret.secret}) -> tensor<4xi16> {
    %c1 = arith.constant 1 : index
    // expected-error@+1 {{--arith-to-cggi does not support tensor_ext operations. Lower them to scalars first.}}
    %0 = tensor_ext.rotate %arg0, %c1 : tensor<4xi16>, index
    return %0 : tensor<4xi16>
  }
}
