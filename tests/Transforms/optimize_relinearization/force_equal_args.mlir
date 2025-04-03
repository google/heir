// RUN: heir-opt --mlir-print-local-scope --secretize --mlir-to-secret-arithmetic --optimize-relinearization='allow-mixed-degree-operands=false' %s | FileCheck %s

// CHECK: func.func @relinearize_both_add_operands
// CHECK: secret.generic
// CHECK: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: tensor_ext.rotate
// CHECK-NEXT: arith.addi
// CHECK-NOT: mgmt.relinearize
// CHECK-NEXT: secret.yield
func.func @relinearize_both_add_operands(%arg0: tensor<8xi16>, %arg1: tensor<8xi16>) -> tensor<8xi16> {
  %0 = arith.muli %arg0, %arg0: tensor<8xi16>
  %1 = mgmt.relinearize %0  : tensor<8xi16>
  %2 = arith.muli %arg1, %arg1: tensor<8xi16>
  %3 = mgmt.relinearize %2  : tensor<8xi16>

  // Rotation requires degree 1 key basis input
  %c1 = arith.constant 1 : index
  %6 = tensor_ext.rotate %3, %c1 : tensor<8xi16>, index
  %7 = arith.addi %1, %6 : tensor<8xi16>
  func.return %7 : tensor<8xi16>
}
