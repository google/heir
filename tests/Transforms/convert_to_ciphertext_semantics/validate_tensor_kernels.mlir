// RUN: heir-opt %s --validate-tensor-kernels --split-input-file --verify-diagnostics

func.func @runtime_index(%x: tensor<4xf32> {secret.secret}, %i: index) -> f32 {
  // expected-error@+2 {{indexing an encrypted tensor with a runtime index is not supported}}
  // expected-note@+1 {{tensor.extract requires compile-time constant indices}}
  %y = tensor.extract %x[%i] : tensor<4xf32>
  return %y : f32
}

// -----

func.func @nonzero_padding(%x: tensor<4xf32> {secret.secret}) -> tensor<6xf32> {
  %one = arith.constant 1.0 : f32
  // expected-error@+2 {{padding an encrypted tensor with a nonzero or runtime value is not supported}}
  // expected-note@+1 {{this lowering supports constant zero padding}}
  %y = tensor.pad %x low[1] high[1] {
  ^bb0(%i: index):
    tensor.yield %one : f32
  } : tensor<4xf32> to tensor<6xf32>
  return %y : tensor<6xf32>
}

// -----

func.func @dynamic_shape(%x: tensor<?xf32> {secret.secret}) -> tensor<?xf32> {
  %zero = arith.constant 0.0 : f32
  // expected-error@+2 {{padding an encrypted tensor requires static input and output shapes}}
  // expected-note@+1 {{export the model with fixed tensor dimensions}}
  %y = tensor.pad %x low[1] high[1] {
  ^bb0(%i: index):
    tensor.yield %zero : f32
  } : tensor<?xf32> to tensor<?xf32>
  return %y : tensor<?xf32>
}

// -----

// Cleartext operations may use runtime indices and nonzero padding.
func.func @cleartext(%x: tensor<4xf32>, %i: index) -> f32 {
  %one = arith.constant 1.0 : f32
  %p = tensor.pad %x low[1] high[1] {
  ^bb0(%j: index):
    tensor.yield %one : f32
  } : tensor<4xf32> to tensor<6xf32>
  %y = tensor.extract %p[%i] : tensor<6xf32>
  return %y : f32
}

func.func @supported(%x: tensor<4xf32> {secret.secret}) -> f32 {
  %zero = arith.constant 0.0 : f32
  %i = arith.constant 2 : index
  %p = tensor.pad %x low[1] high[1] {
  ^bb0(%j: index):
    tensor.yield %zero : f32
  } : tensor<4xf32> to tensor<6xf32>
  %y = tensor.extract %p[%i] : tensor<6xf32>
  return %y : f32
}

// -----

func.func @runtime_padding_value(%x: tensor<4xf32> {secret.secret}, %value: f32) -> tensor<6xf32> {
  // expected-error@+2 {{padding an encrypted tensor with a nonzero or runtime value is not supported}}
  // expected-note@+1 {{this lowering supports constant zero padding}}
  %y = tensor.pad %x low[1] high[1] {
  ^bb0(%i: index):
    tensor.yield %value : f32
  } : tensor<4xf32> to tensor<6xf32>
  return %y : tensor<6xf32>
}

// -----

func.func @runtime_padding_amount(%x: tensor<4xf32> {secret.secret}, %amount: index) -> tensor<6xf32> {
  %zero = arith.constant 0.0 : f32
  // expected-error@+2 {{runtime padding amounts for encrypted tensors are not supported}}
  // expected-note@+1 {{use compile-time constant low and high padding amounts}}
  %y = tensor.pad %x low[%amount] high[1] {
  ^bb0(%i: index):
    tensor.yield %zero : f32
  } : tensor<4xf32> to tensor<6xf32>
  return %y : tensor<6xf32>
}
