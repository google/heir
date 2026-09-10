// RUN: heir-opt --add-client-interface=enable-layout-assignment=false %s | FileCheck %s --implicit-check-not=heir.entry_func --implicit-check-not=server.evaluate_func

// A packing helper is part of the entry's interface, not another entry.
func.func private @pack(%x: tensor<4xf32>) -> tensor<4xf32> attributes {client.pack_func = {func_name = "entry"}} {
  return %x : tensor<4xf32>
}

// CHECK: func.func @entry
// CHECK-SAME: heir.entry_func = {func_name = "entry"}
// CHECK-SAME: server.evaluate_func = {func_name = "entry"}
func.func @entry(%x: tensor<4xf32>) -> tensor<4xf32> {
  %y = func.call @pack(%x) : (tensor<4xf32>) -> tensor<4xf32>
  return %y : tensor<4xf32>
}
