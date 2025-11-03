// RUN: heir-opt --validate-noise=model=bgv-noise-kpz21 %s | FileCheck %s

// CHECK: module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [2148728833, 2148794369, 1152921504607338497], P = [1152921504608747521, 1152921504609239041], plaintextModulus = 65537>, scheme.bgv} {
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [2148728833, 2148794369, 1152921504607338497], P = [1152921504608747521, 1152921504609239041], plaintextModulus = 65537>, scheme.bgv} {
  // CHECK: @return
  func.func @return(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 0>}) -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 0>}) {
    return %arg0 : !secret.secret<tensor<1024xi16>>
  }
}
