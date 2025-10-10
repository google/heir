// RUN: heir-opt --validate-noise=model=bgv-noise-by-bound-coeff-average-case %s | FileCheck %s

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [134250497, 33832961, 140737488486401], P = [140737488928769, 140737489256449], plaintextModulus = 65537>, scheme.bgv} {
  // CHECK: @dot_product
  func.func @dot_product(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}, %arg1: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %c7 = arith.constant 7 : index
    %c1_i16 = arith.constant 1 : i16
    %cst = arith.constant dense<0> : tensor<1024xi16>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %inserted = tensor.insert %c1_i16 into %cst[%c7] : tensor<1024xi16>
    %0 = mgmt.init %inserted {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1024xi16>
    %1 = secret.generic(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}, %arg1: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) {
    ^body(%input0: tensor<1024xi16>, %input1: tensor<1024xi16>):
      %2 = arith.muli %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<1024xi16>
      %3 = mgmt.relinearize %2 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>
      %4 = tensor_ext.rotate %3, %c4 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>, index
      %5 = arith.addi %3, %4 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>
      %6 = tensor_ext.rotate %5, %c2 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>, index
      %7 = arith.addi %5, %6 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>
      %8 = tensor_ext.rotate %7, %c1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>, index
      %9 = arith.addi %7, %8 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>
      %10 = mgmt.modreduce %9 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1024xi16>
      %11 = arith.muli %0, %10 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1024xi16>
      %12 = tensor_ext.rotate %11, %c7 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1024xi16>, index
      %13 = mgmt.modreduce %12 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1024xi16>
      secret.yield %13 : tensor<1024xi16>
    } -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %1 : !secret.secret<tensor<1024xi16>>
  }
}
