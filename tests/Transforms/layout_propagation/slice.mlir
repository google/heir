// RUN: heir-opt --layout-propagation %s -verify-diagnostics

module {
  func.func @test(%arg0: !secret.secret<tensor<16xi8>>) -> !secret.secret<tensor<11x16xi8>> {
    %0 = tensor.empty() : tensor<11x16xi8>
    %1 = secret.generic(%arg0: !secret.secret<tensor<16xi8>>) {
    ^body(%input0: tensor<16xi8>):
    // expected-error @+1 {{Layout propagation not supported for this op}}
      %extracted_slice = tensor.extract_slice %input0[0] [4] [1] : tensor<16xi8> to tensor<4xi8>
      %inserted_slice = tensor.insert_slice %extracted_slice into %0[0, 0] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
      secret.yield %inserted_slice : tensor<11x16xi8>
    } -> !secret.secret<tensor<11x16xi8>>
    return %1 : !secret.secret<tensor<11x16xi8>>
  }
}
