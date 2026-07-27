// RUN: heir-opt --implement-shift-network %s | FileCheck %s

#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (4i0 + 5i1 + slot) mod 30 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 1 and 0 <= slot <= 1023 }">
#replication = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 6 = 0 and 0 <= i1 <= 5 and 0 <= slot <= 1023 }">

module {
  // CHECK: @periodic_replication
  // CHECK-NOT: tensor_ext.remap
  // CHECK: tensor.extract_slice
  // CHECK: tensor_ext.rotate
  // CHECK: tensor.insert_slice
  func.func @periodic_replication(%arg0: tensor<1x1024xi16> {tensor_ext.layout = #layout}) -> (tensor<1x1024xi16> {tensor_ext.layout = #layout}) {
    %0 = tensor_ext.remap %arg0 { permutation = #replication } : tensor<1x1024xi16>
    return %0 : tensor<1x1024xi16>
  }
}
