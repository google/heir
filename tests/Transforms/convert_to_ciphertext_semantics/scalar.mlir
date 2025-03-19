// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=1024 | FileCheck %s

#alignment = #tensor_ext.alignment<in = [], out = [1], insertedDims = [0]>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment>

// CHECK: scalar_mul
// CHECK-COUNT-7: tensor<1024xi16>
func.func @scalar_mul(%arg0: !secret.secret<i16> {tensor_ext.layout = #layout}) -> (!secret.secret<i16> {tensor_ext.layout = #layout}) {
  %0 = secret.generic ins(%arg0 : !secret.secret<i16>) attrs = {__argattrs = [{tensor_ext.layout = #layout}], __resattrs = [{tensor_ext.layout = #layout}]} {
  ^body(%input0: i16):
    %1 = arith.muli %input0, %input0 {tensor_ext.layout = #layout} : i16
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
