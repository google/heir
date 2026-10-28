// RUN: heir-opt --split-input-file --verify-diagnostics --generate-param-ckks="min-slot-count=1024" %s | FileCheck %s

// Tests for generate-param-ckks when the module is pre-annotated with a
// #cheddar.parameter_set attribute instead of using a parameter file.

// A module with only cheddar.parameter_set gets ckks.schemeParam derived from it.
// CHECK: module attributes {
// CHECK-DAG: cheddar.parameter_set = #cheddar.parameter_set<logN = 14, logScale = 30, mainPrimes = [1000000007, 1000000009, 1000000021, 1000000033], auxPrimes = [2000000011, 2000000033], levelConfig = [2, 0, 4, 0], wordBits = 32, defaultEncryptionLevel = 1>
// CHECK-DAG: ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [1000000007, 1000000009, 1000000021, 1000000033], P = [2000000011, 2000000033], logDefaultScale = 30, encryptionTechnique = extended, moduliPerLevel = [2, 2]>
// CHECK-DAG: scheme.actual_slot_count = 8192 : i64
// CHECK-DAG: scheme.requested_slot_count = 1024 : i64
// CHECK: func.func @only_parameter_set

module attributes {backend.cheddar, cheddar.parameter_set = #cheddar.parameter_set<logN = 14, logScale = 30, mainPrimes = [1000000007, 1000000009, 1000000021, 1000000033], auxPrimes = [2000000011, 2000000033], levelConfig = [2, 0, 4, 0], wordBits = 32, defaultEncryptionLevel = 1>} {
  func.func @only_parameter_set(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    return %arg0 : !secret.secret<f16>
  }
}

// -----

// Both attributes present and consistent: the pass leaves them untouched.
// CHECK: module attributes {
// CHECK-DAG: cheddar.parameter_set = #cheddar.parameter_set<logN = 14, logScale = 30, mainPrimes = [1000000007, 1000000009, 1000000021, 1000000033], auxPrimes = [2000000011, 2000000033], levelConfig = [2, 0, 4, 0], wordBits = 32, defaultEncryptionLevel = 1>
// CHECK-DAG: ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [1000000007, 1000000009, 1000000021, 1000000033], P = [2000000011, 2000000033], logDefaultScale = 30, encryptionTechnique = extended, moduliPerLevel = [2, 2]>
// CHECK: func.func @consistent

module attributes {backend.cheddar, cheddar.parameter_set = #cheddar.parameter_set<logN = 14, logScale = 30, mainPrimes = [1000000007, 1000000009, 1000000021, 1000000033], auxPrimes = [2000000011, 2000000033], levelConfig = [2, 0, 4, 0], wordBits = 32, defaultEncryptionLevel = 1>, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [1000000007, 1000000009, 1000000021, 1000000033], P = [2000000011, 2000000033], logDefaultScale = 30, encryptionTechnique = extended, moduliPerLevel = [2, 2]>} {
  func.func @consistent(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    return %arg0 : !secret.secret<f16>
  }
}

// -----

// Both attributes present but inconsistent (logDefaultScale differs).
// expected-error@+1 {{ckks.schemeParam attribute is inconsistent with cheddar.parameter_set attribute}}
module attributes {backend.cheddar, cheddar.parameter_set = #cheddar.parameter_set<logN = 14, logScale = 30, mainPrimes = [1000000007, 1000000009, 1000000021, 1000000033], auxPrimes = [2000000011, 2000000033], levelConfig = [2, 0, 4, 0], wordBits = 32, defaultEncryptionLevel = 1>, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [1000000007, 1000000009, 1000000021, 1000000033], P = [2000000011, 2000000033], logDefaultScale = 40, encryptionTechnique = extended, moduliPerLevel = [2, 2]>} {
  func.func @inconsistent(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    return %arg0 : !secret.secret<f16>
  }
}
