// RUN: heir-opt --validate-preprocessing %s | FileCheck %s

// CHECK: func @multi_type_pairing
func.func @multi_type_pairing(%arg0: i32, %arg1: f32, %arg2: index) -> (i32, f32) {
  %storage = preprocessing.empty : !preprocessing.storage<i32, f32>
  preprocessing.store %arg0, %storage[%arg2] site 0 <i32> : i32, !preprocessing.storage<i32, f32>
  preprocessing.store %arg1, %storage[%arg2] site 1 <f32> : f32, !preprocessing.storage<i32, f32>
  %res0 = preprocessing.load %storage[%arg2] site 0 <i32> : !preprocessing.storage<i32, f32>, i32
  %res1 = preprocessing.load %storage[%arg2] site 1 <f32> : !preprocessing.storage<i32, f32>, f32
  return %res0, %res1 : i32, f32
}
