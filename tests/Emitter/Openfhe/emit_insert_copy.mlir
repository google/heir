// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module attributes {scheme.bgv} {
  // Two tensor.insert ops sharing the same tensor.empty buffer must produce
  // distinct C++ vectors (copy-on-write), not alias both results to the same
  // destination variable.

  // CHECK: mm_bug_two_wraps
  // CHECK: std::vector<{{.*}}> [[V1:v[0-9]+]](
  // CHECK: [[V1]][0] = %arg1
  // CHECK: std::vector<{{.*}}> [[V2:v[0-9]+]](
  // CHECK: [[V2]][0] = %arg2
  // CHECK: return {[[V1]], [[V2]]};
  func.func @mm_bug_two_wraps(%arg0: !cc, %arg1: !ct, %arg2: !ct)
      -> (tensor<1x!ct>, tensor<1x!ct>) {
    %c0 = arith.constant 0 : index
    %buf = tensor.empty() : tensor<1x!ct>
    %wrap1 = tensor.insert %arg1 into %buf[%c0] : tensor<1x!ct>
    %wrap2 = tensor.insert %arg2 into %buf[%c0] : tensor<1x!ct>
    return %wrap1, %wrap2 : tensor<1x!ct>, tensor<1x!ct>
  }
}
