// RUN: heir-translate --emit-openfhe-emitc %s | FileCheck %s --check-prefix=CHECK-IMPL
// RUN: heir-translate --emit-openfhe-pybind %s | FileCheck %s --check-prefix=CHECK-BIND

// CHECK-IMPL: CiphertextT test_function
// CHECK-IMPL: struct return_i32_i32
// CHECK-IMPL-NOT: pybind11
// CHECK-IMPL-NOT: PYBIND11_MODULE

// CHECK-BIND: #include <pybind11/pybind11.h>
// CHECK-BIND: #include <pybind11/stl.h>
// CHECK-BIND: #include "tests/Emitter/OpenfheEmitC/test_lib.h"
// CHECK-BIND: PYBIND11_MODULE(test_bindings, m) {
// CHECK-BIND:   py::class_<[[STRUCT_CT:return_.*]]>(m, "[[STRUCT_CT]]", py::module_local())
// CHECK-BIND:     .def_readwrite("field0", &[[STRUCT_CT]]::field0)
// CHECK-BIND:     .def_readwrite("field1", &[[STRUCT_CT]]::field1)
// CHECK-BIND:   py::class_<return_i32_i32>(m, "return_i32_i32", py::module_local())
// CHECK-BIND:     .def_readwrite("field0", &return_i32_i32::field0)
// CHECK-BIND:     .def_readwrite("field1", &return_i32_i32::field1)
// CHECK-BIND:   m.def("test_function", &test_function, py::call_guard<py::gil_scoped_release>())
// CHECK-BIND:   m.def("test_multi_i32", &test_multi_i32, py::call_guard<py::gil_scoped_release>())
// CHECK-BIND:   m.def("test_multi_return", &test_multi_return, py::call_guard<py::gil_scoped_release>())
// CHECK-BIND: }

module {
  emitc.func public @test_function(%arg0: !emitc.opaque<"CryptoContextT">, %arg1: !emitc.opaque<"CiphertextT">) -> !emitc.opaque<"CiphertextT"> {
    return %arg1 : !emitc.opaque<"CiphertextT">
  }
  emitc.class struct @return_i32_i32 {
    emitc.field @field0 : i32
    emitc.field @field1 : i32
  }
  emitc.func public @test_multi_i32(%arg0: i32) -> !emitc.opaque<"struct return_i32_i32"> {
    %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>
    %1 = "emitc.member"(%0) <{member = "field0"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
    assign %arg0 : i32 to %1 : <i32>
    %2 = "emitc.member"(%0) <{member = "field1"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
    assign %arg0 : i32 to %2 : <i32>
    %3 = load %0 : <!emitc.opaque<"struct return_i32_i32">>
    return %3 : !emitc.opaque<"struct return_i32_i32">
  }
  emitc.class struct @return__emitc_opaque__CiphertextT____emitc_opaque__CiphertextT__ {
    emitc.field @field0 : !emitc.opaque<"CiphertextT">
    emitc.field @field1 : !emitc.opaque<"CiphertextT">
  }
  emitc.func public @test_multi_return(%arg0: !emitc.opaque<"CryptoContextT">, %arg1: !emitc.opaque<"CiphertextT">) -> !emitc.opaque<"struct return__emitc_opaque__CiphertextT____emitc_opaque__CiphertextT__"> {
    %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"struct return__emitc_opaque__CiphertextT____emitc_opaque__CiphertextT__">>
    %1 = "emitc.member"(%0) <{member = "field0"}> : (!emitc.lvalue<!emitc.opaque<"struct return__emitc_opaque__CiphertextT____emitc_opaque__CiphertextT__">>) -> !emitc.lvalue<!emitc.opaque<"CiphertextT">>
    assign %arg1 : !emitc.opaque<"CiphertextT"> to %1 : <!emitc.opaque<"CiphertextT">>
    %2 = "emitc.member"(%0) <{member = "field1"}> : (!emitc.lvalue<!emitc.opaque<"struct return__emitc_opaque__CiphertextT____emitc_opaque__CiphertextT__">>) -> !emitc.lvalue<!emitc.opaque<"CiphertextT">>
    assign %arg1 : !emitc.opaque<"CiphertextT"> to %2 : <!emitc.opaque<"CiphertextT">>
    %3 = load %0 : <!emitc.opaque<"struct return__emitc_opaque__CiphertextT____emitc_opaque__CiphertextT__">>
    return %3 : !emitc.opaque<"struct return__emitc_opaque__CiphertextT____emitc_opaque__CiphertextT__">
  }
  module @pybind_bindings attributes {heir.pybind_module, pybind.imports = ["third_party/heir/tests/Emitter/OpenfheEmitC/test_lib.h"], pybind.module_name = "test_bindings"} {
    emitc.func private @test_function(!emitc.opaque<"CryptoContextT">, !emitc.opaque<"CiphertextT">) -> !emitc.opaque<"CiphertextT"> attributes {specifiers = ["extern"]}
    emitc.class struct @return_i32_i32 {
      emitc.field @field0 : i32
      emitc.field @field1 : i32
    }
    emitc.func private @test_multi_i32(i32) -> !emitc.opaque<"struct return_i32_i32"> attributes {specifiers = ["extern"]}
    emitc.class struct @return__emitc_opaque__CiphertextT____emitc_opaque__CiphertextT__ {
      emitc.field @field0 : !emitc.opaque<"CiphertextT">
      emitc.field @field1 : !emitc.opaque<"CiphertextT">
    }
    emitc.func private @test_multi_return(!emitc.opaque<"CryptoContextT">, !emitc.opaque<"CiphertextT">) -> !emitc.opaque<"struct return__emitc_opaque__CiphertextT____emitc_opaque__CiphertextT__"> attributes {specifiers = ["extern"]}
  }
}
