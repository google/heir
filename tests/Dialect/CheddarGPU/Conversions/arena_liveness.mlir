// RUN: heir-opt %s --cheddar-gpu-to-gpu | FileCheck %s

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 4, ter = [], main = [974258177], aux = [2147352577], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {

  // Check arena size attribute and global alloc
  // CHECK: func.func private @__cheddar_gpu_init
  // CHECK: gpu.alloc{{.*}}: memref<512xi8>

  // Test (a): Two sequential non-overlapping temporaries reuse the same offset 0.
  // CHECK: func.func @test_sequential_reuse
  // CHECK: %[[C0_A:.*]] = arith.constant 0 : index
  // CHECK: memref.view %{{.*}}[%[[C0_A]]][] : memref<512xi8> to memref<4x16xi32>
  // CHECK: gpu.launch_func
  // CHECK: %[[C0_B:.*]] = arith.constant 0 : index
  // CHECK: memref.view %{{.*}}[%[[C0_B]]][] : memref<512xi8> to memref<4x16xi32>
  // CHECK: gpu.launch_func
  func.func @test_sequential_reuse(%arg0: memref<4x16xi32>) {
    %t0 = memref.alloc() : memref<4x16xi32>
    cheddar_gpu.add %t0, %arg0, %arg0 <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x16xi32>
    %t1 = memref.alloc() : memref<4x16xi32>
    cheddar_gpu.add %t1, %arg0, %arg0 <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x16xi32>
    return
  }

  // Test (b): A temporary kept alive across alias/regions is NOT reused inside its live range.
  // CHECK: func.func @test_live_across_region
  // CHECK: %[[C0_REG:.*]] = arith.constant 0 : index
  // CHECK: memref.view %{{.*}}[%[[C0_REG]]][] : memref<512xi8> to memref<4x16xi32>
  // CHECK: scf.if
  // CHECK: %[[C256_INNER:.*]] = arith.constant 256 : index
  // CHECK: memref.view %{{.*}}[%[[C256_INNER]]][] : memref<512xi8> to memref<4x16xi32>
  // CHECK: gpu.launch_func
  // CHECK: gpu.launch_func
  func.func @test_live_across_region(%cond: i1, %arg0: memref<4x16xi32>, %out: memref<4x16xi32>) {
    %t0 = memref.alloc() : memref<4x16xi32>
    %sub = memref.subview %t0[0, 0] [2, 16] [1, 1] : memref<4x16xi32> to memref<2x16xi32>
    scf.if %cond {
      %inner = memref.alloc() : memref<4x16xi32>
      cheddar_gpu.add %inner, %arg0, %arg0 <{
        lhs_form = #cheddar_gpu.form<eval>,
        rhs_form = #cheddar_gpu.form<eval>,
        out_form = #cheddar_gpu.form<eval>
      }> : memref<4x16xi32>
    }
    cheddar_gpu.add %out, %t0, %arg0 <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x16xi32>
    return
  }

  // Test (c): Two root entry functions in one module do NOT sum (arena = max over roots).
  // CHECK: func.func @root_a
  // CHECK: %[[C0_ROOT_A:.*]] = arith.constant 0 : index
  // CHECK: memref.view %{{.*}}[%[[C0_ROOT_A]]][] : memref<512xi8> to memref<4x16xi32>
  func.func @root_a(%arg0: memref<4x16xi32>) {
    %a = memref.alloc() : memref<4x16xi32>
    cheddar_gpu.add %a, %arg0, %arg0 <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x16xi32>
    return
  }

  // CHECK: func.func @root_b
  // CHECK: %[[C0_ROOT_B:.*]] = arith.constant 0 : index
  // CHECK: memref.view %{{.*}}[%[[C0_ROOT_B]]][] : memref<512xi8> to memref<8x16xi32>
  func.func @root_b(%arg0: memref<8x16xi32>) {
    %b = memref.alloc() : memref<8x16xi32>
    cheddar_gpu.add %b, %arg0, %arg0 <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<8x16xi32>
    return
  }

  // Test (d): @main -> @helper via func.call get disjoint offset ranges.
  // CHECK: func.func @caller
  // CHECK: %[[C0_CALLER:.*]] = arith.constant 0 : index
  // CHECK: memref.view %{{.*}}[%[[C0_CALLER]]][] : memref<512xi8> to memref<4x16xi32>
  // CHECK: call @callee
  func.func @caller(%arg0: memref<4x16xi32>) {
    %c = memref.alloc() : memref<4x16xi32>
    cheddar_gpu.add %c, %arg0, %arg0 <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x16xi32>
    func.call @callee(%arg0) : (memref<4x16xi32>) -> ()
    cheddar_gpu.add %c, %arg0, %arg0 <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x16xi32>
    return
  }

  // CHECK: func.func private @callee
  // CHECK: %[[C256_CALLEE:.*]] = arith.constant 256 : index
  // CHECK: memref.view %{{.*}}[%[[C256_CALLEE]]][] : memref<512xi8> to memref<4x16xi32>
  func.func private @callee(%arg0: memref<4x16xi32>) {
    %h = memref.alloc() : memref<4x16xi32>
    cheddar_gpu.add %h, %arg0, %arg0 <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x16xi32>
    return
  }
}
