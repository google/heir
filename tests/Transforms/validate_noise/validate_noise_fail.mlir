// RUN: heir-opt --validate-noise="model=bgv-noise-by-bound-coeff-worst-case" --verify-diagnostics %s

// This is only for testing whether validate-noise would fail, but
// not for testing the noise model.
// This failing example is handcrafted to fail the noise validation
// using the following conditions:
//   + plaintext modulus is large
//   + modulus is 45 bits
//   + consecutive multiplications
//   + using worst-case noise model
//   + When plaintext modulus is large, modulus of 45 bits is not enough
// Note that if any condition changed the test may fail and changes
// to this file are expected

// expected-error@below {{Noise validation failed.}}
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113, 35184376184833], P = [35184376545281, 35184376578049], plaintextModulus = 4295294977>, scheme.bgv} {
  func.func @dot_product(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 5, scale = 0>}) -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 5, scale = 0>}) {
    ^body(%input0: tensor<1024xi16>):
      %1 = arith.muli %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 5, dimension = 3, scale = 0>} : tensor<1024xi16>
      %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 5, scale = 0>} : tensor<1024xi16>
      %3 = mgmt.modreduce %2 {mgmt.mgmt = #mgmt.mgmt<level = 4, scale = 0>} : tensor<1024xi16>
      %4 = arith.muli %3, %3 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3, scale = 0>} : tensor<1024xi16>
      %5 = mgmt.relinearize %4 {mgmt.mgmt = #mgmt.mgmt<level = 4, scale = 0>} : tensor<1024xi16>
      %6 = mgmt.modreduce %5 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 0>} : tensor<1024xi16>
      %7 = arith.muli %6, %6 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3, scale = 0>} : tensor<1024xi16>
      %8 = mgmt.relinearize %7 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 0>} : tensor<1024xi16>
      %9 = mgmt.modreduce %8 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 0>} : tensor<1024xi16>
      %10 = arith.muli %9, %9 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3, scale = 0>} : tensor<1024xi16>
      %11 = mgmt.relinearize %10 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 0>} : tensor<1024xi16>
      %12 = mgmt.modreduce %11 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 0>} : tensor<1024xi16>
      %13 = arith.muli %12, %12 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3, scale = 0>} : tensor<1024xi16>
      %14 = mgmt.relinearize %13 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 0>} : tensor<1024xi16>
      %15 = mgmt.modreduce %14 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>} : tensor<1024xi16>
      secret.yield %15 : tensor<1024xi16>
    } -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>})
    return %0 : !secret.secret<tensor<1024xi16>>
  }
}
