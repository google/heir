// RUN: heir-opt --add-client-interface=enable-layout-assignment=false %s | FileCheck %s --implicit-check-not='roles = ["entry"' --implicit-check-not='"server.evaluate"'

// A packing helper is part of the entry's interface, not another entry.
// Updating an existing interface retains metadata and does not duplicate roles.
func.func private @pack(%x: tensor<4xf32>) -> tensor<4xf32> attributes {heir.interface = {func_name = "entry", roles = ["client.pack"]}} {
  return %x : tensor<4xf32>
}

// CHECK: func.func @entry
// CHECK-SAME: heir.interface = {extra = "keep", func_name = "entry", input_types = [tensor<4xf32>], result_types = [tensor<4xf32>], roles = ["entry", "server.evaluate"]}
func.func @entry(%x: tensor<4xf32>) -> tensor<4xf32> attributes {heir.interface = {extra = "keep", func_name = "entry", roles = ["entry", "server.evaluate"]}} {
  %y = func.call @pack(%x) : (tensor<4xf32>) -> tensor<4xf32>
  return %y : tensor<4xf32>
}
