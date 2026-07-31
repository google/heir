// RUN: heir-opt --verify-diagnostics --split-input-file %s

// The emitters compute the resource size with ShapedType::getNumElements(),
// which asserts unless the shape is static, so reject non-static shapes here
// rather than crashing in the backend.

func.func @dynamic_dim() -> memref<?xi32> {
  // expected-error@+1 {{must have a static shape}}
  %0 = preprocessing.load_resource "p/dyn.bin" : memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

func.func @unranked() -> tensor<*xi32> {
  // expected-error@+1 {{must have a static shape}}
  %0 = preprocessing.load_resource "p/unranked.bin" : tensor<*xi32>
  return %0 : tensor<*xi32>
}

// -----

// A static shape is accepted.
func.func @static_ok() -> tensor<4xi32> {
  %0 = preprocessing.load_resource "p/static.bin" : tensor<4xi32>
  return %0 : tensor<4xi32>
}
