// RUN: heir-opt --convert-to-emitc="lower-to-cpp=false" %s | FileCheck %s

// CHECK: module {
// CHECK:   emitc.func public @test_function(%{{.*}}: !emitc.opaque<"CryptoContextT">, %[[ARG1:.*]]: !emitc.opaque<"CiphertextT">) -> !emitc.opaque<"CiphertextT"> {
// CHECK:     return %[[ARG1]] : !emitc.opaque<"CiphertextT">
// CHECK:   }

// CHECK:   emitc.class struct @[[STRUCT_I32:.*]] {
// CHECK:     emitc.field @field0 : i32
// CHECK:     emitc.field @field1 : i32
// CHECK:   }
// CHECK:   emitc.func public @test_multi_i32(%[[ARG_I32:.*]]: i32) -> !emitc.opaque<"struct [[STRUCT_I32]]">

// CHECK:   emitc.class struct @[[STRUCT_CT:.*]] {
// CHECK:     emitc.field @field0 : !emitc.opaque<"CiphertextT">
// CHECK:     emitc.field @field1 : !emitc.opaque<"CiphertextT">
// CHECK:   }
// CHECK:   emitc.func public @test_multi_return(%{{.*}}: !emitc.opaque<"CryptoContextT">, %[[ARG2:.*]]: !emitc.opaque<"CiphertextT">) -> !emitc.opaque<"struct [[STRUCT_CT]]">

// CHECK:   module @pybind_bindings attributes {heir.pybind_module} {
// CHECK:     emitc.func private @test_function(!emitc.opaque<"CryptoContextT">, !emitc.opaque<"CiphertextT">) -> !emitc.opaque<"CiphertextT">
// CHECK:     emitc.class struct @[[STRUCT_I32]] {
// CHECK:       emitc.field @field0 : i32
// CHECK:       emitc.field @field1 : i32
// CHECK:     }
// CHECK:     emitc.func private @test_multi_i32(i32) -> !emitc.opaque<"struct [[STRUCT_I32]]">
// CHECK:     emitc.class struct @[[STRUCT_CT]] {
// CHECK:       emitc.field @field0 : !emitc.opaque<"CiphertextT">
// CHECK:       emitc.field @field1 : !emitc.opaque<"CiphertextT">
// CHECK:     }
// CHECK:     emitc.func private @test_multi_return(!emitc.opaque<"CryptoContextT">, !emitc.opaque<"CiphertextT">) -> !emitc.opaque<"struct [[STRUCT_CT]]">
// CHECK:   }
// CHECK: }

module {
  func.func public @test_function(%arg0: !openfhe.crypto_context, %arg1: !openfhe.ciphertext) -> !openfhe.ciphertext {
    return %arg1 : !openfhe.ciphertext
  }
  func.func public @test_multi_i32(%arg0: i32) -> (i32, i32) {
    return %arg0, %arg0 : i32, i32
  }
  func.func public @test_multi_return(%arg0: !openfhe.crypto_context, %arg1: !openfhe.ciphertext) -> (!openfhe.ciphertext, !openfhe.ciphertext) {
    return %arg1, %arg1 : !openfhe.ciphertext, !openfhe.ciphertext
  }
  module @pybind_bindings attributes {heir.pybind_module} {
    func.func private @test_function(!openfhe.crypto_context, !openfhe.ciphertext) -> !openfhe.ciphertext
    func.func private @test_multi_i32(i32) -> (i32, i32)
    func.func private @test_multi_return(!openfhe.crypto_context, !openfhe.ciphertext) -> (!openfhe.ciphertext, !openfhe.ciphertext)
  }
}
