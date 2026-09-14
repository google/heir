// RUN: heir-opt %s --torch-linalg-to-ckks --verify-diagnostics

// Exercise the public pipeline, rather than only the standalone validator.
func.func @runtime_index(%x: tensor<4xf32> {secret.secret}, %i: index) -> f32 {
  // expected-error@+2 {{indexing an encrypted tensor with a runtime index is not supported}}
  // expected-note@+1 {{tensor.extract requires compile-time constant indices}}
  %y = tensor.extract %x[%i] : tensor<4xf32>
  return %y : f32
}
