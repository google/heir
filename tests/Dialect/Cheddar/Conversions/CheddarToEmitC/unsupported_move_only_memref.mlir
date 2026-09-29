// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar --split-input-file --verify-diagnostics %s

// A dynamic-shape payload memref has no fixed-size C++ representation.
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @dynamic(%m: memref<?x!cheddar.ciphertext>) {
  return
}

// -----

// Non-unit payload strides cannot be represented by a C array.
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @strided_payload(
    %m: memref<4x!cheddar.ciphertext, strided<[2]>>) {
  return
}

// -----

// Likewise, a raw float pointer would lose this stride and miscompile users.
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @strided_float(%ctx: !cheddar.context,
                         %m: memref<4xf32, strided<[2]>>) {
  return
}

// -----

// A payload subview must select a single element.
func.func @drop_middle_dimension(
    %m: memref<2x1x4x!cheddar.ciphertext>) {
  // expected-error @below {{failed to legalize operation 'memref.subview'}}
  %slice = memref.subview %m[0, 0, 0] [2, 1, 4] [1, 1, 1]
      : memref<2x1x4x!cheddar.ciphertext> to memref<2x4x!cheddar.ciphertext>
  return
}
