// RUN: heir-opt --softmax-to-cgf-softmax %s 2>&1 | FileCheck %s

// CHECK: warning: input domain width ({{.*}}3{{.*}}) exceeds the recommended safe limit (2.0) for CGF-softmax approximation. Accuracy may degrade.
// CHECK: func @softmax_warning
func.func @softmax_warning(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NOT: math_ext.softmax
  %0 = math_ext.softmax %arg0 {domain_lower = -2.0 : f64, domain_upper = 1.0 : f64} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
