// RUN: heir-opt --validate-preprocessing --verify-diagnostics --split-input-file %s

func.func @multiple_allocations() {
  // expected-note@+1 {{previous allocation here}}
  %storage1 = preprocessing.empty : !preprocessing.storage<i32>
  // expected-error@+1 {{more than one preprocessing.empty allocation in function}}
  %storage2 = preprocessing.empty : !preprocessing.storage<i32>
  return
}

// -----

func.func @missing_load(%arg0: i32, %arg1: index) {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  // expected-error@+1 {{lacks a corresponding load for site_id: 0}}
  preprocessing.store %arg0, %storage[%arg1] site 0 <i32> : i32, !preprocessing.storage<i32>
  return
}

// -----

func.func @missing_store(%arg1: index) -> i32 {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  // expected-error@+1 {{lacks a corresponding store for site_id: 0}}
  %res = preprocessing.load %storage[%arg1] site 0 <i32> : !preprocessing.storage<i32>, i32
  return %res : i32
}

// -----

func.func @duplicate_store(%arg0: i32, %arg1: index) -> i32 {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  // expected-note@+1 {{previous store here}}
  preprocessing.store %arg0, %storage[%arg1] site 0 <i32> : i32, !preprocessing.storage<i32>
  // expected-error@+1 {{duplicate store for site_id: 0}}
  preprocessing.store %arg0, %storage[%arg1] site 0 <i32> : i32, !preprocessing.storage<i32>
  %res = preprocessing.load %storage[%arg1] site 0 <i32> : !preprocessing.storage<i32>, i32
  return %res : i32
}

// -----

func.func @duplicate_load(%arg0: i32, %arg1: index) -> i32 {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  preprocessing.store %arg0, %storage[%arg1] site 0 <i32> : i32, !preprocessing.storage<i32>
  // expected-note@+1 {{previous load here}}
  %res1 = preprocessing.load %storage[%arg1] site 0 <i32> : !preprocessing.storage<i32>, i32
  // expected-error@+1 {{duplicate load for site_id: 0}}
  %res2 = preprocessing.load %storage[%arg1] site 0 <i32> : !preprocessing.storage<i32>, i32
  return %res2 : i32
}

// -----

func.func @mismatched_arity(%arg0: i32, %arg1: index, %arg2: index) -> i32 {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  // expected-error@+1 {{store and load index arity mismatch for site_id: 0}}
  preprocessing.store %arg0, %storage[%arg1, %arg2] site 0 <i32> : i32, !preprocessing.storage<i32>
  // expected-note@+1 {{corresponding load here}}
  %res = preprocessing.load %storage[%arg1] site 0 <i32> : !preprocessing.storage<i32>, i32
  return %res : i32
}

// -----

func.func @mismatched_stored_type(%arg0: i64, %arg1: index) {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  // expected-error@+1 {{stored value type 'i64' does not match op element type 'i32'}}
  preprocessing.store %arg0, %storage[%arg1] site 0 <i32> : i64, !preprocessing.storage<i32>
  %res = preprocessing.load %storage[%arg1] site 0 <i32> : !preprocessing.storage<i32>, i32
  return
}

// -----

func.func @mismatched_loaded_type(%arg0: i32, %arg1: index) {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  preprocessing.store %arg0, %storage[%arg1] site 0 <i32> : i32, !preprocessing.storage<i32>
  // expected-error@+1 {{loaded value type 'i64' does not match op element type 'i32'}}
  %res = preprocessing.load %storage[%arg1] site 0 <i32> : !preprocessing.storage<i32>, i64
  return
}

// -----

func.func @stored_type_not_in_storage(%arg0: i64, %arg1: index) {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  // expected-error@+1 {{op element type 'i64' is not in storage element types}}
  preprocessing.store %arg0, %storage[%arg1] site 0 <i64> : i64, !preprocessing.storage<i32>
  %res = preprocessing.load %storage[%arg1] site 0 <i32> : !preprocessing.storage<i32>, i32
  return
}

// -----

func.func @loaded_type_not_in_storage(%arg0: i32, %arg1: index) {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  preprocessing.store %arg0, %storage[%arg1] site 0 <i32> : i32, !preprocessing.storage<i32>
  // expected-error@+1 {{op element type 'i64' is not in storage element types}}
  %res = preprocessing.load %storage[%arg1] site 0 <i64> : !preprocessing.storage<i32>, i64
  return
}

// -----

func.func @mismatched_op_element_type(%arg0: i32, %arg1: index) {
  %storage = preprocessing.empty : !preprocessing.storage<i32, f32>
  // expected-error@+1 {{store element type 'i32' does not match load element type 'f32'}}
  preprocessing.store %arg0, %storage[%arg1] site 0 <i32> : i32, !preprocessing.storage<i32, f32>
  // expected-note@+1 {{corresponding load here}}
  %res = preprocessing.load %storage[%arg1] site 0 <f32> : !preprocessing.storage<i32, f32>, f32
  return
}

// -----

func.func @mismatched_storage_type(%arg0: i32, %arg1: index, %storage1: !preprocessing.storage<i32>, %storage2: !preprocessing.storage<i32, f32>) {
  // expected-error@+1 {{store storage type '!preprocessing.storage<i32>' does not match load storage type '!preprocessing.storage<i32, f32>'}}
  preprocessing.store %arg0, %storage1[%arg1] site 0 <i32> : i32, !preprocessing.storage<i32>
  // expected-note@+1 {{corresponding load here}}
  %res = preprocessing.load %storage2[%arg1] site 0 <i32> : !preprocessing.storage<i32, f32>, i32
  return
}

// -----

// Multiple allocations are allowed if they are in different functions.
func.func @multiple_allocations_ok_1() {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  return
}

func.func @multiple_allocations_ok_2() {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  return
}
