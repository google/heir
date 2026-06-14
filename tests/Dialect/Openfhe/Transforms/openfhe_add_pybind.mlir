// RUN: heir-opt --openfhe-add-pybind="pybind-module-name=test_bindings pybind-imports=third_party/heir/tests/Dialect/Openfhe/Transforms/test_lib.h" %s | FileCheck %s

// CHECK: ![[CT:.*]] = !openfhe.ciphertext

// CHECK: module {
// CHECK:   func.func public @test_function(%[[ARG0:.*]]: ![[CT]]) -> ![[CT]] {
// CHECK:     return %[[ARG0]] : ![[CT]]
// CHECK:   }
// CHECK:   module @pybind_bindings attributes {heir.pybind_module, pybind.imports = ["third_party/heir/tests/Dialect/Openfhe/Transforms/test_lib.h"], pybind.module_name = "test_bindings"} {
// CHECK:     func.func private @test_function(![[CT]]) -> ![[CT]]
// CHECK:   }
// CHECK: }

module {
  func.func public @test_function(%arg0: !openfhe.ciphertext) -> !openfhe.ciphertext {
    return %arg0 : !openfhe.ciphertext
  }
}
