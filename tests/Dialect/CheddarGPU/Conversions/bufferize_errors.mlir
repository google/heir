// RUN: heir-opt %s --one-shot-bufferize="bufferize-function-boundaries" --verify-diagnostics

#eval = #cheddar_gpu.form<eval>

// Buffers in different memory spaces cannot be cast to a common type.
func.func @test_add_memory_space_mismatch(%lhs_buffer: memref<4x65536xi32, 1>, %rhs: tensor<4x65536xi32>) -> tensor<4x65536xi32> {
  %lhs = bufferization.to_tensor %lhs_buffer restrict : memref<4x65536xi32, 1> to tensor<4x65536xi32>
  %empty = tensor.empty() : tensor<4x65536xi32>
  // expected-error @below {{cannot bufferize: operand buffers are in different memory spaces}}
  // expected-error @below {{failed to bufferize op}}
  %0 = cheddar_gpu.add %empty, %lhs, %rhs <{
    lhs_form = #eval,
    rhs_form = #eval,
    out_form = #eval
  }> : tensor<4x65536xi32> -> tensor<4x65536xi32>
  return %0 : tensor<4x65536xi32>
}
