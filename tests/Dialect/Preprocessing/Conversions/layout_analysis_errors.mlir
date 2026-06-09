// RUN: heir-opt --preprocessing-to-memref --verify-diagnostics --split-input-file %s

func.func @duplicate_store(%arg0: i32) {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  preprocessing.store %arg0, %storage[] site 0 <i32> : i32, !preprocessing.storage<i32>
  // expected-error@+1 {{Duplicate StoreOp found for site_id: 0}}
  preprocessing.store %arg0, %storage[] site 0 <i32> : i32, !preprocessing.storage<i32>
  return
}

// -----

func.func @missing_store(%storage: !preprocessing.storage<i32>) {
  // expected-error@+1 {{LoadOp found with site_id 0 but no corresponding StoreOp exists}}
  %res = preprocessing.load %storage[] site 0 <i32> : !preprocessing.storage<i32>, i32
  return
}

// -----

func.func @index_arity_mismatch(%arg0: i32, %arg1: index) {
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  // expected-error@+2 {{Number of indices (1) does not match number of enclosing loops (0)}}
  // expected-error@+1 {{failed to legalize operation 'preprocessing.store' that was explicitly marked illegal}}
  preprocessing.store %arg0, %storage[%arg1] site 0 <i32> : i32, !preprocessing.storage<i32>
  return
}
