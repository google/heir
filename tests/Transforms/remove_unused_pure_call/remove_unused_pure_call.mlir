// RUN: heir-opt --remove-unused-pure-call %s | FileCheck %s

// CHECK: module
module {
  func.func @pure_func(%arg0: i32) -> i32 attributes {client.pack_func} {
    return %arg0 : i32
  }

  func.func @not_pure_func(%arg0: i32) -> i32 {
    return %arg0 : i32
  }

  func.func @pure_multi_res(%arg0: i32) -> (i32, i32) attributes {client.pack_func} {
    return %arg0, %arg0 : i32, i32
  }

  // CHECK: func @test_remove_unused
  // CHECK-SAME: (%[[ARG0:.*]]: i32)
  func.func @test_remove_unused(%arg0: i32) {
    // CHECK-NOT: call @pure_func
    %0 = func.call @pure_func(%arg0) : (i32) -> i32
    return
  }

  // CHECK: func @test_remove_multi_unused
  // CHECK-SAME: (%[[ARG0:.*]]: i32)
  func.func @test_remove_multi_unused(%arg0: i32) {
    // CHECK-NOT: call @pure_multi_res
    %0, %1 = func.call @pure_multi_res(%arg0) : (i32) -> (i32, i32)
    return
  }

  // CHECK: func @test_keep_used
  // CHECK-SAME: (%[[ARG0:.*]]: i32)
  func.func @test_keep_used(%arg0: i32) -> i32 {
    // CHECK: %[[RES:.*]] = call @pure_func(%[[ARG0]])
    // CHECK: return %[[RES]]
    %0 = func.call @pure_func(%arg0) : (i32) -> i32
    return %0 : i32
  }

  // CHECK: func @test_keep_multi_used
  // CHECK-SAME: (%[[ARG0:.*]]: i32)
  func.func @test_keep_multi_used(%arg0: i32) -> i32 {
    // CHECK: %[[RES:.*]]:2 = call @pure_multi_res(%[[ARG0]])
    // CHECK: return %[[RES]]#0
    %0, %1 = func.call @pure_multi_res(%arg0) : (i32) -> (i32, i32)
    return %0 : i32
  }

  // CHECK: func @test_keep_not_pure
  // CHECK-SAME: (%[[ARG0:.*]]: i32)
  func.func @test_keep_not_pure(%arg0: i32) {
    // CHECK: call @not_pure_func(%[[ARG0]])
    %0 = func.call @not_pure_func(%arg0) : (i32) -> i32
    return
  }
}

// Role metadata does not make writes, including nested writes, removable.
func.func @setup(%out: memref<1xi32>) attributes {client.setup_func = {func_name = "read_initialized"}} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  memref.store %c1, %out[%c0] : memref<1xi32>
  return
}

func.func @pack(%out: memref<1xi32>, %condition: i1) attributes {client.pack_func} {
  scf.if %condition {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : i32
    memref.store %c2, %out[%c0] : memref<1xi32>
  }
  return
}

// CHECK: func.func @read_initialized
// CHECK: call @setup
// CHECK: call @pack
// CHECK: memref.load
func.func @read_initialized(%out: memref<1xi32>, %condition: i1) -> i32 {
  func.call @setup(%out) : (memref<1xi32>) -> ()
  func.call @pack(%out, %condition) : (memref<1xi32>, i1) -> ()
  %c0 = arith.constant 0 : index
  %value = memref.load %out[%c0] : memref<1xi32>
  return %value : i32
}

func.func private @external_helper(i32) -> i32 attributes {client.pack_func}

// CHECK: func.func @keep_unknown_effects
// CHECK: call @external_helper
func.func @keep_unknown_effects(%x: i32) {
  %unused = func.call @external_helper(%x) : (i32) -> i32
  return
}
