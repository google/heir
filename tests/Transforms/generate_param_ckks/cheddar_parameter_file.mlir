// RUN: heir-opt --generate-param-ckks="parameter-file=%S/cheddar_params.json min-slot-count=1024" %s | FileCheck %s
// RUN: not heir-opt --generate-param-ckks="parameter-file=%S/cheddar_params.json min-slot-count=16384" %s 2>&1 | FileCheck %s --check-prefix=ERR-SLOT
// RUN: sed 's/level = 0/level = 3/' %s | not heir-opt --generate-param-ckks="parameter-file=%S/cheddar_params.json min-slot-count=1024" 2>&1 | FileCheck %s --check-prefix=ERR-LEVEL
// RUN: sed 's/level = 0/level = 2/' %s | not heir-opt --generate-param-ckks="parameter-file=%S/cheddar_params.json min-slot-count=1024" 2>&1 | FileCheck %s --check-prefix=ERR-ENC-LEVEL
// RUN: sed 's/{backend.cheddar}/{backend.cheddar, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [1000000007], P = [2000000011], logDefaultScale = 30>}/' %s | not heir-opt --generate-param-ckks="parameter-file=%S/cheddar_params.json min-slot-count=1024" 2>&1 | FileCheck %s --check-prefix=ERR-HAS-SCHEME-PARAM
// RUN: sed 's/{backend.cheddar}/{backend.cheddar, cheddar.parameter_set = #cheddar.parameter_set<logN = 14, logScale = 30, mainPrimes = [1000000007], auxPrimes = [2000000011]>}/' %s | not heir-opt --generate-param-ckks="parameter-file=%S/cheddar_params.json min-slot-count=1024" 2>&1 | FileCheck %s --check-prefix=ERR-HAS-PARAM-SET

// CHECK: module attributes {
// CHECK-DAG: backend.cheddar
// CHECK-DAG: cheddar.parameter_set = #cheddar.parameter_set<logN = 14, logScale = 30, mainPrimes = [1000000007, 1000000009, 1000000021, 1000000033, 1000000087, 1000000093], auxPrimes = [2000000011, 2000000033], levelConfig = [2, 0, 4, 0, 6, 0], wordBits = 32, defaultEncryptionLevel = 1>
// CHECK-DAG: ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [1000000007, 1000000009, 1000000021, 1000000033, 1000000087, 1000000093], P = [2000000011, 2000000033], logDefaultScale = 30, encryptionTechnique = extended, moduliPerLevel = [2, 2, 2]>
// CHECK-DAG: scheme.actual_slot_count = 8192 : i64
// CHECK-DAG: scheme.requested_slot_count = 1024 : i64
// CHECK: func.func @add

// ERR-SLOT: The slot count in the scheme param (8192) is smaller than the minimum slot count (16384).
// ERR-LEVEL: The level in the scheme param is smaller than the max level.
// ERR-ENC-LEVEL: The max level (2) exceeds the Cheddar defaultEncryptionLevel (1); levels above it are reserved for bootstrapping.
// ERR-HAS-SCHEME-PARAM: cannot specify parameter-file when module already has ckks.schemeParam attribute
// ERR-HAS-PARAM-SET: cannot specify parameter-file when module already has cheddar.parameter_set attribute

module attributes {backend.cheddar} {
  func.func @add(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: f16):
      %1 = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f16
      secret.yield %1 : f16
    } -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<f16>
  }
}
