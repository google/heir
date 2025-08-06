// RUN: heir-opt --new-layout-propagation=ciphertext-size=1024 %s | FileCheck %s

// Test that layout propagation can handle secret scalars

// CHECK: #tensor_ext.new_layout<domainSize=0, relation="(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 1023 >= 0)">

// CHECK: @scalar_mul
func.func @scalar_mul(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
  %0 = secret.generic(%arg0 : !secret.secret<i16>) {
  ^body(%input0: i16):
    %1 = arith.muli %input0, %input0: i16
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
