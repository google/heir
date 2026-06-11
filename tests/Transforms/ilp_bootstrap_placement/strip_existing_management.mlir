// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=3 %s | FileCheck %s

// The ILP pass recomputes management placement, so pre-existing management ops
// in a secret.generic body should be removed before solving.

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

module attributes {scheme.ckks} {
  // CHECK: func.func @strips_existing_management
  // CHECK: secret.generic
  // CHECK: arith.addf %input0, %input0
  // CHECK-NOT: id = 77
  // CHECK-NOT: mgmt.modreduce
  // CHECK-NOT: mgmt.level_reduce
  // CHECK-NOT: mgmt.bootstrap
  // CHECK-NOT: mgmt.relinearize
  // CHECK: secret.yield
  func.func @strips_existing_management(%arg0: !ct_ty) -> !ct_ty {
    %0 = secret.generic(%arg0: !ct_ty) {
    ^body(%input0: !pt_ty):
      %old_adjust = mgmt.adjust_scale %input0 {id = 77 : i64} : !pt_ty
      %old_modreduce = mgmt.modreduce %old_adjust : !pt_ty
      %old_level_reduce = mgmt.level_reduce %old_modreduce
          {levelToDrop = 2 : i64} : !pt_ty
      %old_bootstrap = mgmt.bootstrap %old_level_reduce : !pt_ty
      %old_relinearize = mgmt.relinearize %old_bootstrap : !pt_ty
      %out = arith.addf %old_relinearize, %input0 : !pt_ty
      secret.yield %out : !pt_ty
    } -> !ct_ty
    return %0 : !ct_ty
  }
}
