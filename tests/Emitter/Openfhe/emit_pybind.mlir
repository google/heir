// RUN: heir-translate %s --emit-openfhe-pke-pybind --pybind-header-include=foo.h --pybind-module-name=_heir_foo | FileCheck %s

// CHECK: #include <pybind11/pybind11.h>
// CHECK: #include <pybind11/stl.h>

// A minor hack to avoid copybara mangling this transformation when it is synced
// internally to Google.
// CHECK: #include
// CHECK-SAME: "foo.h"

// CHECK: using namespace lbcrypto;
// CHECK: namespace py = pybind11;
// CHECK: void bind_common(py::module &m)
// CHECK: {
// CHECK:    py::class_<PublicKeyImpl<DCRTPoly>, std::shared_ptr<PublicKeyImpl<DCRTPoly>>>(m, "PublicKey", py::module_local())
// CHECK:        .def(py::init<>());
// CHECK:    py::class_<PrivateKeyImpl<DCRTPoly>, std::shared_ptr<PrivateKeyImpl<DCRTPoly>>>(m, "PrivateKey", py::module_local())
// CHECK:        .def(py::init<>());
// CHECK:    py::class_<KeyPair<DCRTPoly>>(m, "KeyPair", py::module_local())
// CHECK:        .def_readwrite("publicKey", &KeyPair<DCRTPoly>::publicKey)
// CHECK:        .def_readwrite("secretKey", &KeyPair<DCRTPoly>::secretKey);
// CHECK:    py::class_<CiphertextImpl<DCRTPoly>, std::shared_ptr<CiphertextImpl<DCRTPoly>>>(m, "Ciphertext", py::module_local())
// CHECK:        .def(py::init<>());
// CHECK:    py::class_<CryptoContextImpl<DCRTPoly>, std::shared_ptr<CryptoContextImpl<DCRTPoly>>>(m, "CryptoContext", py::module_local())
// CHECK:        .def(py::init<>())
// CHECK:        .def("KeyGen", &CryptoContextImpl<DCRTPoly>::KeyGen);
// CHECK: }

// CHECK: PYBIND11_MODULE(_heir_foo, m) {
// CHECK:   bind_common(m);
// CHECK:   m.def("simple_sum", &simple_sum, py::call_guard<py::gil_scoped_release>());
// CHECK:   m.def("simple_sum__encrypt", &simple_sum__encrypt, py::call_guard<py::gil_scoped_release>());
// CHECK:   m.def("simple_sum__decrypt", &simple_sum__decrypt, py::call_guard<py::gil_scoped_release>());
// CHECK: }

!cc = !openfhe.crypto_context
!ek = !openfhe.eval_key
!ct = !openfhe.ciphertext
!pt = !openfhe.plaintext

func.func @simple_sum(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  %1 = openfhe.rot %arg0, %arg1 { index = 16 } : (!openfhe.crypto_context, !ct) -> !ct
  %2 = openfhe.add %arg0, %arg1, %1 : (!openfhe.crypto_context, !ct, !ct) -> !ct
  %4 = openfhe.rot %arg0, %2 { index = 8 } : (!openfhe.crypto_context, !ct) -> !ct
  %5 = openfhe.add %arg0, %2, %4 : (!openfhe.crypto_context, !ct, !ct) -> !ct
  %7 = openfhe.rot %arg0, %5 { index = 4 } : (!openfhe.crypto_context, !ct) -> !ct
  %8 = openfhe.add %arg0, %5, %7 : (!openfhe.crypto_context, !ct, !ct) -> !ct
  %10 = openfhe.rot %arg0, %8 { index = 2 } : (!openfhe.crypto_context, !ct) -> !ct
  %11 = openfhe.add %arg0, %8, %10 : (!openfhe.crypto_context, !ct, !ct) -> !ct
  %13 = openfhe.rot %arg0, %11 { index = 1 } : (!openfhe.crypto_context, !ct) -> !ct
  %14 = openfhe.add %arg0, %11, %13 : (!openfhe.crypto_context, !ct, !ct) -> !ct
  %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi16>
  %15 = openfhe.make_packed_plaintext %arg0, %cst : (!openfhe.crypto_context, tensor<32xi16>) -> !pt
  %16 = openfhe.mul_plain %arg0, %14, %15 : (!openfhe.crypto_context, !ct, !pt) -> !ct
  %18 = openfhe.rot %arg0, %16 { index = 31 } : (!openfhe.crypto_context, !ct) -> !ct
  return %18 : !ct
}
func.func @simple_sum__encrypt(%arg0: !openfhe.crypto_context, %arg1: tensor<32xi16>, %arg2: !openfhe.public_key) -> !ct {
  %0 = openfhe.make_packed_plaintext %arg0, %arg1 : (!openfhe.crypto_context, tensor<32xi16>) -> !pt
  %1 = openfhe.encrypt %arg0, %0, %arg2 : (!openfhe.crypto_context, !pt, !openfhe.public_key) -> !ct
  return %1 : !ct
}
func.func @simple_sum__decrypt(%arg0: !openfhe.crypto_context, %arg1: !ct, %arg2: !openfhe.private_key) -> i16 {
  %0 = openfhe.decrypt %arg0, %arg1, %arg2 : (!openfhe.crypto_context, !ct, !openfhe.private_key) -> !pt
  %1 = openfhe.decode %0 : !pt -> i16
  return %1 : i16
}
