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

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>

!cc = !openfhe.crypto_context
!ek = !openfhe.eval_key

!tensor_pt_ty = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space>
!scalar_pt_ty = !lwe.new_lwe_plaintext<application_data = <message_type = i16>, plaintext_space = #plaintext_space>
!tensor_ct_ty = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!scalar_ct_ty = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

func.func @simple_sum(%arg0: !openfhe.crypto_context, %arg1: !tensor_ct_ty) -> !scalar_ct_ty {
  %1 = openfhe.rot %arg0, %arg1 { index = 16 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %2 = openfhe.add %arg0, %arg1, %1 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
  %4 = openfhe.rot %arg0, %2 { index = 8 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %5 = openfhe.add %arg0, %2, %4 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
  %7 = openfhe.rot %arg0, %5 { index = 4 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %8 = openfhe.add %arg0, %5, %7 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
  %10 = openfhe.rot %arg0, %8 { index = 2 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %11 = openfhe.add %arg0, %8, %10 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
  %13 = openfhe.rot %arg0, %11 { index = 1 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %14 = openfhe.add %arg0, %11, %13 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
  %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi16>
  %15 = openfhe.make_packed_plaintext %arg0, %cst : (!openfhe.crypto_context, tensor<32xi16>) -> !tensor_pt_ty
  %16 = openfhe.mul_plain %arg0, %14, %15 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_pt_ty) -> !tensor_ct_ty
  %18 = openfhe.rot %arg0, %16 { index = 31 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %19 = lwe.reinterpret_application_data %18 : !tensor_ct_ty to !scalar_ct_ty
  return %19 : !scalar_ct_ty
}
func.func @simple_sum__encrypt(%arg0: !openfhe.crypto_context, %arg1: tensor<32xi16>, %arg2: !openfhe.public_key) -> !tensor_ct_ty {
  %0 = openfhe.make_packed_plaintext %arg0, %arg1 : (!openfhe.crypto_context, tensor<32xi16>) -> !tensor_pt_ty
  %1 = openfhe.encrypt %arg0, %0, %arg2 : (!openfhe.crypto_context, !tensor_pt_ty, !openfhe.public_key) -> !tensor_ct_ty
  return %1 : !tensor_ct_ty
}
func.func @simple_sum__decrypt(%arg0: !openfhe.crypto_context, %arg1: !scalar_ct_ty, %arg2: !openfhe.private_key) -> i16 {
  %0 = openfhe.decrypt %arg0, %arg1, %arg2 : (!openfhe.crypto_context, !scalar_ct_ty, !openfhe.private_key) -> !scalar_pt_ty
  %1 = lwe.rlwe_decode %0 {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : !scalar_pt_ty -> i16
  return %1 : i16
}
