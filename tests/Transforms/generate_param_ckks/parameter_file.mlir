// RUN: heir-opt --generate-param-ckks="parameter-file=%S/custom_params.yaml min-slot-count=1024" %s | FileCheck %s
// RUN: heir-opt --generate-param-ckks="parameter-file=%S/custom_params.json min-slot-count=1024" %s | FileCheck %s
// RUN: not heir-opt --generate-param-ckks="parameter-file=%S/custom_params.yaml min-slot-count=4096" %s 2>&1 | FileCheck %s --check-prefix=ERR-SLOT
// RUN: not heir-opt --generate-param-ckks="parameter-file=%S/non_existent.yaml" %s 2>&1 | FileCheck %s --check-prefix=ERR-FILE
// RUN: sed 's/level = 0/level = 2/' %s | not heir-opt --generate-param-ckks="parameter-file=%S/custom_params.yaml min-slot-count=1024" 2>&1 | FileCheck %s --check-prefix=ERR-LEVEL
// RUN: sed 's/module {/module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 12, Q = [553648129, 70368744161281, 70368744177665, 70368744210433], P = [1152921504606994433], logDefaultScale = 45>} {/' %s | not heir-opt --generate-param-ckks="parameter-file=%S/custom_params.yaml" 2>&1 | FileCheck %s --check-prefix=ERR-EXISTING

// CHECK: module attributes {
// CHECK-DAG: ckks.schemeParam = #ckks.scheme_param<logN = 12, Q = [553648129, 70368744161281, 70368744177665, 70368744210433], P = [1152921504606994433], logDefaultScale = 45, encryptionType = sk, encryptionTechnique = extended, moduliPerLevel = [2, 2]>
// CHECK-DAG: scheme.actual_slot_count = 2048 : i64
// CHECK-DAG: scheme.requested_slot_count = 1024 : i64
// CHECK: func.func @add

// ERR-SLOT: The slot count in the scheme param (2048) is smaller than the minimum slot count (4096).
// ERR-FILE: cannot read CKKS parameter file
// ERR-LEVEL: The level in the scheme param is smaller than the max level.
// ERR-EXISTING: cannot specify parameter-file when module already has ckks.scheme_param attribute

module {
  func.func @add(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: f16):
      %1 = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f16
      secret.yield %1 : f16
    } -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<f16>
  }
}
