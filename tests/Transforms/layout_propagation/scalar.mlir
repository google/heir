// RUN: heir-opt --layout-propagation=ciphertext-size=1024 %s | FileCheck %s

// Test that layout propagation can handle secret scalars

// CHECK: #tensor_ext.alignment<in = [], out = [1024], insertedDims = [0]>
// CHECK: #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment>

// CHECK: @scalar_mul
func.func @scalar_mul(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
  %0 = secret.generic(%arg0 : !secret.secret<i16>) {
  ^body(%input0: i16):
    %1 = arith.muli %input0, %input0: i16
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
