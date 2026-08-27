// RUN: heir-opt --debug-insert-linalg-checkpoints="entry-function=main" --debug-insert-linalg-checkpoints="entry-function=main" %s | FileCheck %s

#identity = affine_map<(d0, d1) -> (d0, d1)>

module {
  // CHECK: func.func @main
  func.func @main(%secret: tensor<2x2xf32> {secret.secret},
                  %public: tensor<2x2xf32>) -> tensor<2x2xf32> {
    // CHECK: debug.validate %arg0 {name = "main/input/0"}
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<2x2xf32>
    // This cleartext initializer is not a checkpoint.
    // CHECK: %[[FILL:.*]] = linalg.fill
    // CHECK-NEXT: %[[MATMUL:.*]] = linalg.matmul
    %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    %matmul = linalg.matmul ins(%secret, %public : tensor<2x2xf32>, tensor<2x2xf32>) outs(%fill : tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK-NEXT: debug.validate %[[MATMUL]] {name = "main/linalg.matmul/1/0"}
    // CHECK: %[[GENERIC:.*]] = linalg.generic
    %generic = linalg.generic {
        indexing_maps = [#identity, #identity],
        iterator_types = ["parallel", "parallel"]}
        ins(%matmul : tensor<2x2xf32>)
        outs(%empty : tensor<2x2xf32>) {
      ^bb0(%in: f32, %out: f32):
        %sum = arith.addf %in, %in : f32
        linalg.yield %sum : f32
    } -> tensor<2x2xf32>
    // CHECK: debug.validate %[[GENERIC]] {name = "main/linalg.generic/2/0"}
    return %generic : tensor<2x2xf32>
  }

  // `entry-function` prevents sibling model/helper functions from being
  // instrumented.
  // CHECK: func.func @other
  // CHECK-NOT: debug.validate
  func.func @other(%secret: tensor<2x2xf32> {secret.secret},
                   %public: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %empty = tensor.empty() : tensor<2x2xf32>
    %matmul = linalg.matmul ins(%secret, %public : tensor<2x2xf32>, tensor<2x2xf32>) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    return %matmul : tensor<2x2xf32>
  }
}
