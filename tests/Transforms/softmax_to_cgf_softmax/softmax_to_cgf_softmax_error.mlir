// RUN: not heir-opt --softmax-to-cgf-softmax %s 2>&1 | FileCheck %s

// CHECK: error: 'math_ext.softmax' op input domain width ({{.*}}5{{.*}}) exceeds the maximum safe limit (4.0) for CGF-softmax approximation
func.func @softmax_error(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = math_ext.softmax %arg0 {domain_lower = -3.0 : f64, domain_upper = 2.0 : f64} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
