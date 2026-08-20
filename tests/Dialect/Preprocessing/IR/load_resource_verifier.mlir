// RUN: heir-opt --verify-diagnostics --split-input-file %s

// Resource emitters use getNumElements(), which requires a static shape.
func.func @dynamic_dim(%destination: memref<?xi32>) {
  // expected-error@+1 {{must have a static shape}}
  preprocessing.load_resource "p/dyn.bin" into %destination
      : (memref<?xi32>) -> ()
  return
}

// -----

func.func @unranked(%destination: tensor<*xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{must have a static shape}}
  %0 = preprocessing.load_resource "p/unranked.bin" into %destination
      : (tensor<*xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
}

// -----

// A static shape is accepted.
func.func @static_ok() -> tensor<4xi32> {
  %destination = tensor.empty() : tensor<4xi32>
  %0 = preprocessing.load_resource "p/static.bin" into %destination
      : (tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}
