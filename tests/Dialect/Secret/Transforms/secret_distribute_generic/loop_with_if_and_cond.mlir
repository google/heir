// RUN: heir-opt --secret-distribute-generic %s | FileCheck %s

// This test was extracted and minimized from a rolled matvec kernel, the main
// issue was that when collapsing a secret generic, the distribute-generic pass
// would re-conceal the result value from the collapsed generic, but not
// further merge the conceal into downstream generics (using
// ConcealThenGeneric). For distributing through an if statement, this resulted
// in the condition of the if being treated as secret. So this test basically
// ensures that ConcealThenGeneric is kept inside the secret-distribute-generic
// pass.
func.func @test_distribute_through_if_with_cond(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c23 = arith.constant 23 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant dense<2.0> : tensor<1x1024xf32>
  %0 = arith.constant dense<5.0> : tensor<512x1024xf32>
  %8 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>, %arg0: !secret.secret<tensor<1x1024xf32>>) {
  ^body(%input0: tensor<1x1024xf32>, %input1: tensor<1x1024xf32>):
    %14 = scf.for %arg1 = %c1 to %c23 step %c2 iter_args(%arg2 = %input0) -> (tensor<1x1024xf32>) {
      %15 = mgmt.bootstrap %arg2 : tensor<1x1024xf32>

      // CHECK:      [[v:%.*]] = arith.cmpi slt, {{.*}} : index
      // CHECK-NEXT: scf.if [[v]] -> (!secret.secret<tensor<1x1024xf32>>) {
      %16 = arith.cmpi slt, %arg1, %c512 : index
      %17 = scf.if %16 -> (tensor<1x1024xf32>) {
        %extracted_slice_1 = tensor.extract_slice %0[%arg1, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1x1024xf32>
        %21 = tensor_ext.rotate %input1, %arg1 : tensor<1x1024xf32>, index
        %23 = arith.mulf %extracted_slice_1, %21 : tensor<1x1024xf32>
        %24 = mgmt.modreduce %23 : tensor<1x1024xf32>
        %26 = arith.mulf %15, %cst : tensor<1x1024xf32>
        %27 = mgmt.modreduce %26 : tensor<1x1024xf32>
        %28 = arith.addf %27, %24 : tensor<1x1024xf32>
        scf.yield %28 : tensor<1x1024xf32>
      } else {
        scf.yield %15 : tensor<1x1024xf32>
      }
      scf.yield %17 : tensor<1x1024xf32>
    }
    secret.yield %14 : tensor<1x1024xf32>
  } -> !secret.secret<tensor<1x1024xf32>>

  return %8 : !secret.secret<tensor<1x1024xf32>>
}
