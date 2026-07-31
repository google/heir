// RUN: heir-opt --secret-insert-mgmt-ckks=level-budget=41 %s | FileCheck %s

// Ensure that bootstrapping is not applied to the secret tensor in the loop.

module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: func.func @test_lenet_slice_loop
  // CHECK: scf.for
  // CHECK-NOT: mgmt.bootstrap
  // CHECK: scf.yield
  func.func @test_lenet_slice_loop(%ct_input: !secret.secret<tensor<1x8192xf32>>) -> !secret.secret<tensor<12x8192xf32>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index

    %res = secret.generic(%ct_input: !secret.secret<tensor<1x8192xf32>>) {
    ^body(%input: tensor<1x8192xf32>):
      %empty = tensor.empty() : tensor<12x8192xf32>

      %1 = arith.mulf %input, %input : tensor<1x8192xf32>
      %2 = arith.mulf %1, %1 : tensor<1x8192xf32>
      %3 = arith.mulf %2, %2 : tensor<1x8192xf32>
      %4 = arith.mulf %3, %3 : tensor<1x8192xf32>
      %5 = arith.mulf %4, %4 : tensor<1x8192xf32>
      %6 = arith.mulf %5, %5 : tensor<1x8192xf32>
      %7 = arith.mulf %6, %6 : tensor<1x8192xf32>
      %8 = arith.mulf %7, %7 : tensor<1x8192xf32>
      %9 = arith.mulf %8, %8 : tensor<1x8192xf32>
      %10 = arith.mulf %9, %9 : tensor<1x8192xf32>
      %11 = arith.mulf %10, %10 : tensor<1x8192xf32>
      %12 = arith.mulf %11, %11 : tensor<1x8192xf32>
      %13 = arith.mulf %12, %12 : tensor<1x8192xf32>
      %14 = arith.mulf %13, %13 : tensor<1x8192xf32>
      %15 = arith.mulf %14, %14 : tensor<1x8192xf32>
      %16 = arith.mulf %15, %15 : tensor<1x8192xf32>
      %17 = arith.mulf %16, %16 : tensor<1x8192xf32>
      %18 = arith.mulf %17, %17 : tensor<1x8192xf32>
      %19 = arith.mulf %18, %18 : tensor<1x8192xf32>
      %20 = arith.mulf %19, %19 : tensor<1x8192xf32>
      %21 = arith.mulf %20, %20 : tensor<1x8192xf32>
      %22 = arith.mulf %21, %21 : tensor<1x8192xf32>
      %23 = arith.mulf %22, %22 : tensor<1x8192xf32>
      %24 = arith.mulf %23, %23 : tensor<1x8192xf32>
      %25 = arith.mulf %24, %24 : tensor<1x8192xf32>
      %26 = arith.mulf %25, %25 : tensor<1x8192xf32>
      %27 = arith.mulf %26, %26 : tensor<1x8192xf32>
      %28 = arith.mulf %27, %27 : tensor<1x8192xf32>
      %29 = arith.mulf %28, %28 : tensor<1x8192xf32>
      %30 = arith.mulf %29, %29 : tensor<1x8192xf32>
      %31 = arith.mulf %30, %30 : tensor<1x8192xf32>
      %32 = arith.mulf %31, %31 : tensor<1x8192xf32>
      %33 = arith.mulf %32, %32 : tensor<1x8192xf32>
      %34 = arith.mulf %33, %33 : tensor<1x8192xf32>
      %35 = arith.mulf %34, %34 : tensor<1x8192xf32>
      %36 = arith.mulf %35, %35 : tensor<1x8192xf32>
      %37 = arith.mulf %36, %36 : tensor<1x8192xf32>
      %38 = arith.mulf %37, %37 : tensor<1x8192xf32>
      %39 = arith.mulf %38, %38 : tensor<1x8192xf32>
      %40 = arith.mulf %39, %39 : tensor<1x8192xf32>
      %41 = arith.mulf %40, %40 : tensor<1x8192xf32>

      %loop = scf.for %idx = %c0 to %c12 step %c1 iter_args(%acc = %empty) -> (tensor<12x8192xf32>) {
        %rot = tensor_ext.rotate %41, %idx : tensor<1x8192xf32>, index
        %inserted = tensor.insert_slice %rot into %acc[%idx, 0] [1, 8192] [1, 1] : tensor<1x8192xf32> into tensor<12x8192xf32>
        scf.yield %inserted : tensor<12x8192xf32>
      }
      secret.yield %loop : tensor<12x8192xf32>
    } -> !secret.secret<tensor<12x8192xf32>>

    return %res : !secret.secret<tensor<12x8192xf32>>
  }
}
