// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=2048 | FileCheck %s

// Just testing that the codegen doesn't hang

#layout_reproducer = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : (2048 + 98i0 - 100i1 - i2 + ct + 2048*floor((-98 - 98i0 + slot)/2048)) mod 4096 = 0 and 0 <= i0 <= 15 and 0 <= i1 <= 21 and 0 <= i2 <= 1 and 0 <= ct <= 2047 and 0 <= slot <= 2146 and 2048*floor((-98 - 98i0 + slot)/2048) >= -3615 + slot and 2048*floor((-98 - 98i0 + slot)/2048) >= -2147 - 98i0 + i2 + slot and 2048*floor((-98 - 98i0 + slot)/2048) <= -2048 - 98i0 + slot and 2048*floor((-98 - 98i0 + slot)/2048) <= -2048 - 98i0 + i2 + slot and 2048*floor((-98 - 98i0 + slot)/2048) <= -2048 + slot }">

module {
  func.func @test(%arg0: !secret.secret<tensor<16x22x2xf32>>, %arg1: tensor<16x22x2xf32>) -> (!secret.secret<tensor<16x22x2xf32>> {tensor_ext.layout = #layout_reproducer}) {
    %0 = secret.generic(%arg0 : !secret.secret<tensor<16x22x2xf32>>) {
    ^body(%unused: tensor<16x22x2xf32>):
      %val = tensor_ext.assign_layout %arg1 {domainSchedule = array<i64: 0, 1>, layout = #layout_reproducer, tensor_ext.layout = #layout_reproducer} : tensor<16x22x2xf32>
      secret.yield %val : tensor<16x22x2xf32>
    } -> (!secret.secret<tensor<16x22x2xf32>> {tensor_ext.layout = #layout_reproducer})
    return %0 : !secret.secret<tensor<16x22x2xf32>>
  }
}
// CHECK: func.func @test
