// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=1024 --split-input-file | FileCheck %s

#new_layout = #tensor_ext.new_layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 1023 }">
// CHECK: scalar_mul
// CHECK-SAME: tensor<1x1024xi16>
module {
  func.func @scalar_mul(%arg0: !secret.secret<i16> {tensor_ext.layout = #new_layout}) -> (!secret.secret<i16> {tensor_ext.layout = #new_layout}) {
    %0 = secret.generic(%arg0: !secret.secret<i16> {tensor_ext.layout = #new_layout}) {
    ^body(%input0: i16):
      %1 = arith.muli %input0, %input0 {tensor_ext.layout = #new_layout} : i16
      secret.yield %1 : i16
    } -> (!secret.secret<i16> {tensor_ext.layout = #new_layout})
    return %0 : !secret.secret<i16>
  }
}
