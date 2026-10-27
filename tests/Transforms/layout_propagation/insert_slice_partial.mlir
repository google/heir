// RUN: heir-opt --layout-propagation=min-slot-count=16 %s | FileCheck %s

// A concatenation inserts slices smaller than the result along a non-unit
// dimension. The result keeps a layout of its own shape (here the default),
// rather than the layout of the first slice, which covers only half of it.

// CHECK: [[L:#layout[0-9]*]] = #tensor_ext.layout<"{{.*}}0 <= i1 <= 7{{.*}}">
// CHECK: func.func @concat
// CHECK: tensor_ext.assign_layout %{{.*}} <layout = [[L]]> {{.*}}: tensor<1x8xf32>
func.func @concat(%a: !secret.secret<tensor<1x4xf32>>, %b: !secret.secret<tensor<1x4xf32>>) -> !secret.secret<tensor<1x8xf32>> {
  %empty = tensor.empty() : tensor<1x8xf32>
  %0 = secret.generic(%a: !secret.secret<tensor<1x4xf32>>, %b: !secret.secret<tensor<1x4xf32>>) {
  ^body(%x: tensor<1x4xf32>, %y: tensor<1x4xf32>):
    %1 = tensor.insert_slice %x into %empty[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<1x8xf32>
    %2 = tensor.insert_slice %y into %1[0, 4] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<1x8xf32>
    secret.yield %2 : tensor<1x8xf32>
  } -> !secret.secret<tensor<1x8xf32>>
  return %0 : !secret.secret<tensor<1x8xf32>>
}
