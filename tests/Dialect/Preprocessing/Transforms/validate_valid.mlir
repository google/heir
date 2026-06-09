// RUN: heir-opt --validate-preprocessing %s | FileCheck %s

// CHECK: func @valid_pairing
func.func @valid_pairing(%arg0: i32, %arg1: index) -> i32 {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  preprocessing.store %arg0, %storage[%arg1] site 0 <i32> : i32, !preprocessing.storage<i32>
  %res = preprocessing.load %storage[%arg1] site 0 <i32> : !preprocessing.storage<i32>, i32
  return %res : i32
}
