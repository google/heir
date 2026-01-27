// RUN: heir-translate %s --emit-openfhe-pke --split-input-file | FileCheck %s

// CHECK: struct mnist__preprocessingStruct {
// CHECK:   std::vector<Plaintext> arg0;
// CHECK:   std::vector<Plaintext> arg1;
// CHECK: };
// CHECK: mnist__preprocessingStruct mnist__preprocessing(CryptoContextT [[CC_1:.*]], std::vector<float> [[V0_1:.*]]) {

// CHECK: std::vector<CiphertextT> mnist__preprocessed(CryptoContextT [[CC_2:.*]], std::vector<CiphertextT> [[V0_2:.*]], std::vector<Plaintext> [[V1_2:.*]], std::vector<Plaintext> [[V2_2:.*]]) {

// CHECK: std::vector<CiphertextT> mnist(CryptoContextT [[CC_0:.*]], std::vector<float> [[V0_0:.*]], std::vector<CiphertextT> [[V1_0:.*]]) {
// CHECK:   auto [[V2STRUCT_0:.*]] = mnist__preprocessing([[CC_0]], [[V0_0]]);
// CHECK:   const auto& [[V2_0:.*]] = [[V2STRUCT_0]].arg0;
// CHECK:   const auto& [[V3_0:.*]] = [[V2STRUCT_0]].arg1;
// CHECK:   const auto& [[V4_0:.*]] = mnist__preprocessed([[CC_0]], [[V1_0]], [[V2_0]], [[V3_0]]);
// CHECK:   return [[V4_0]];
// CHECK: }

!cc = !openfhe.crypto_context
!pt = !openfhe.plaintext
!pk = !openfhe.public_key
!ct = !openfhe.ciphertext

module attributes {scheme.ckks} {
  func.func @mnist__preprocessing(%cc: !cc, %arg0: tensor<1024xf32>) -> (tensor<1x!pt>, tensor<512x!pt>) {
    %cst = arith.constant dense<1.0> : tensor<1024xf32>
    %pt = openfhe.make_ckks_packed_plaintext %cc, %cst : (!cc, tensor<1024xf32>) -> !pt
    %from_elements = tensor.from_elements %pt : tensor<1x!pt>
    %pt2 = openfhe.make_ckks_packed_plaintext %cc, %arg0 : (!cc, tensor<1024xf32>) -> !pt
    %splat = tensor.splat %pt2 : tensor<512x!pt>
    return %from_elements, %splat : tensor<1x!pt>, tensor<512x!pt>
  }
  func.func @mnist__preprocessed(%cc: !cc, %arg4: tensor<1x!ct>, %arg5: tensor<1x!pt>, %arg6: tensor<512x!pt>) -> tensor<1x!ct> {
    %c0 = arith.constant 0 : index
    %extracted_ct = tensor.extract %arg4[%c0] : tensor<1x!ct>
    %extracted_pt = tensor.extract %arg5[%c0] : tensor<1x!pt>
    %4 = openfhe.add_plain %cc, %extracted_ct, %extracted_pt : (!cc, !ct, !pt) -> !ct
    %from_elements = tensor.from_elements %4 : tensor<1x!ct>
    return %from_elements : tensor<1x!ct>
  }
  func.func @mnist(%cc: !cc, %arg0: tensor<1024xf32>, %arg4: tensor<1x!ct>) -> (tensor<1x!ct>) {
      %0:2 = call @mnist__preprocessing(%cc, %arg0) : (!cc, tensor<1024xf32>) -> (tensor<1x!pt>, tensor<512x!pt>)
      %1 = call @mnist__preprocessed(%cc, %arg4, %0#0, %0#1) : (!cc, tensor<1x!ct>, tensor<1x!pt>, tensor<512x!pt>) -> tensor<1x!ct>
      return %1 : tensor<1x!ct>
  }
}
