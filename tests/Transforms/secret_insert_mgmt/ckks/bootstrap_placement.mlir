// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-ckks="bootstrap-waterline=2 after-mul=false" %s | FileCheck %s --check-prefix=BEFORE
// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-ckks="bootstrap-waterline=2 after-mul=true" %s | FileCheck %s --check-prefix=AFTER
// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-ckks="bootstrap-waterline=2 level-budget=1 after-mul=true" %s | FileCheck %s --check-prefix=BUDGET

// BEFORE: func.func @bootstrap_placement
// BEFORE: secret.generic
// BEFORE: %[[v0:.*]] = arith.mulf
// BEFORE: %[[v1:.*]] = mgmt.relinearize %[[v0]]
// BEFORE: %[[v2:.*]] = mgmt.modreduce %[[v1]]
// BEFORE: %[[v3:.*]] = arith.mulf %[[v2]], %[[v2]]
// BEFORE: %[[v4:.*]] = mgmt.relinearize %[[v3]]
// BEFORE: %[[v5:.*]] = mgmt.modreduce %[[v4]]
// BEFORE-NOT: mgmt.bootstrap

// AFTER: func.func @bootstrap_placement
// AFTER: secret.generic
// AFTER: %[[v0:.*]] = arith.mulf
// AFTER: %[[v1:.*]] = mgmt.relinearize %[[v0]]
// AFTER: %[[v2:.*]] = mgmt.modreduce %[[v1]]
// AFTER: %[[v3:.*]] = arith.mulf %[[v2]], %[[v2]]
// AFTER: %[[v4:.*]] = mgmt.relinearize %[[v3]]
// AFTER: %[[v5:.*]] = mgmt.modreduce %[[v4]]
// AFTER-NOT: mgmt.bootstrap

// BUDGET: func.func @bootstrap_placement
// BUDGET: secret.generic
// BUDGET: %[[v0:.*]] = arith.mulf
// BUDGET: %[[v1:.*]] = mgmt.relinearize %[[v0]]
// BUDGET: %[[v2:.*]] = mgmt.modreduce %[[v1]]
// BUDGET: %[[v3:.*]] = arith.mulf %[[v2]], %[[v2]]
// BUDGET: %[[v4:.*]] = mgmt.relinearize %[[v3]]
// BUDGET: %[[boot:.*]] = mgmt.bootstrap %[[v4]]
// BUDGET: %[[adj:.*]] = mgmt.adjust_scale %[[boot]] {id = 0 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1>}
// BUDGET: mgmt.modreduce %[[adj]]

func.func @bootstrap_placement(
    %x : f16 {secret.secret}
  ) -> f16 {
    %0 = arith.mulf %x, %x : f16
    %1 = arith.mulf %0, %0 : f16
    return %1 : f16
}
