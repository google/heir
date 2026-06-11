// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=3 scale-waterline=40 scale-factor-bits=51" %s | FileCheck %s

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

module attributes {scheme.ckks} {
  // CHECK: func.func @uses_annotated_input_and_output_scale
  // CHECK: secret.generic
  // CHECK: arith.mulf
  // CHECK: mgmt.modreduce
  func.func @uses_annotated_input_and_output_scale(
      %arg0: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 51>})
      -> (!ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 51>}) {
    %0 = secret.generic(
        %arg0: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 51>}) {
    ^body(%input0: !pt_ty):
      %l1 = arith.mulf %input0, %input0 : !pt_ty
      secret.yield %l1 : !pt_ty
    } -> (!ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 51>})
    return %0 : !ct_ty
  }

  // CHECK: func.func @adds_mismatched_annotated_input_scales
  // CHECK: secret.generic
  // CHECK: mgmt.adjust_scale
  // CHECK: arith.addf
  func.func @adds_mismatched_annotated_input_scales(
      %arg0: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 40>},
      %arg1: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 51>}) -> !ct_ty {
    %0 = secret.generic(
        %arg0: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 40>},
        %arg1: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 51>}) {
    ^body(%input0: !pt_ty, %input1: !pt_ty):
      %out = arith.addf %input0, %input1 : !pt_ty
      secret.yield %out : !pt_ty
    } -> !ct_ty
    return %0 : !ct_ty
  }

  // CHECK: func.func @multiplies_mismatched_annotated_input_scales
  // CHECK: secret.generic
  // CHECK: arith.mulf
  // CHECK: mgmt.relinearize
  // CHECK: mgmt.adjust_scale
  func.func @multiplies_mismatched_annotated_input_scales(
      %arg0: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 40>},
      %arg1: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 51>}) -> !ct_ty {
    %0 = secret.generic(
        %arg0: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 40>},
        %arg1: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 51>}) {
    ^body(%input0: !pt_ty, %input1: !pt_ty):
      %out = arith.mulf %input0, %input1 : !pt_ty
      secret.yield %out : !pt_ty
    } -> !ct_ty
    return %0 : !ct_ty
  }
}
