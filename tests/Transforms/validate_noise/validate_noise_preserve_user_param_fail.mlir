// RUN: heir-opt --validate-noise=model=bgv-noise-kpz21 %s --verify-diagnostics --split-input-file

// expected-error@below {{'builtin.module' op The level in the scheme param is smaller than the max level.}}
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 11, Q = [17], P = [1093633], plaintextModulus = 65537>} {
  func.func @main(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 1>}) -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 1>}) {
    ^body(%input0: tensor<1024xi16>):
      %13 = arith.muli %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<1024xi16>
      %14 = mgmt.relinearize %13 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1024xi16>
      %15 = mgmt.modreduce %14 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1024xi16>
      secret.yield %15 : tensor<1024xi16>
    } -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<tensor<1024xi16>>
  }
}

// -----

// expected-error@below {{Noise validation failed.}}
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 11, Q = [17, 23], P = [1093633], plaintextModulus = 65537>} {
  func.func @main(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 1>}) -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 1>}) {
    ^body(%input0: tensor<1024xi16>):
      %13 = arith.muli %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<1024xi16>
      %14 = mgmt.relinearize %13 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1024xi16>
      %15 = mgmt.modreduce %14 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1024xi16>
      secret.yield %15 : tensor<1024xi16>
    } -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<tensor<1024xi16>>
  }
}
