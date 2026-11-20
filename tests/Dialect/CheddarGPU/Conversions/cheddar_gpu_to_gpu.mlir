// RUN: heir-opt %s --cheddar-gpu-to-gpu --split-input-file | FileCheck %s

// CHECK-NOT: cheddar_gpu.add
// CHECK-NOT: cheddar_gpu.sub

#eval = #cheddar_gpu.form<eval>
#coeff = #cheddar_gpu.form<coeff>
#params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [17], main = [97, 193], aux = [257], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>

module attributes {cheddar_gpu.params = #params} {

  // CHECK: gpu.module @cheddar_kernels attributes {dlti.dl_spec = #dlti.dl_spec<
  // CHECK:   memref.global "private" constant @__cheddar_primes : memref<4xi32> = dense<[17, 97, 193, 257]>
  // CHECK:   gpu.func @cheddar_gpu_add_w32_deg65536
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?x65536xi32>, %[[ARG1:.*]]: memref<?x65536xi32>, %[[ARG2:.*]]: memref<?x65536xi32>,
  // CHECK-SAME:  %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32) kernel {
  // CHECK:     %[[BX_IDX:.*]] = gpu.block_id x
  // CHECK:     %[[TX_IDX:.*]] = gpu.thread_id x
  // CHECK:     %[[BDIM_IDX:.*]] = gpu.block_dim x
  // CHECK:     %[[BX:.*]] = arith.index_cast %[[BX_IDX]] : index to i32
  // CHECK:     %[[TX:.*]] = arith.index_cast %[[TX_IDX]] : index to i32
  // CHECK:     %[[BDIM:.*]] = arith.index_cast %[[BDIM_IDX]] : index to i32
  // CHECK:     %[[BX_DIM:.*]] = arith.muli %[[BX]], %[[BDIM]] : i32
  // CHECK:     %[[TID:.*]] = arith.addi %[[BX_DIM]], %[[TX]] : i32
  // CHECK:     %[[LOCAL_LIMB:.*]] = arith.shrui %[[TID]], %{{.*}} : i32
  // CHECK:     %[[COEFF:.*]] = arith.andi %[[TID]], %{{.*}} : i32
  // CHECK:     %[[IS_AUX:.*]] = arith.cmpi sge, %[[LOCAL_LIMB]], %[[ARG4]] : i32
  // CHECK:     %[[Q_ABS_LIMB:.*]] = arith.addi %[[LOCAL_LIMB]], %[[ARG3]] : i32
  // CHECK:     %[[AUX_OFFSET:.*]] = arith.subi %[[LOCAL_LIMB]], %[[ARG4]] : i32
  // CHECK:     %[[AUX_ABS_LIMB:.*]] = arith.addi %[[AUX_OFFSET]], %[[ARG5]] : i32
  // CHECK:     %[[ABS_LIMB:.*]] = arith.select %[[IS_AUX]], %[[AUX_ABS_LIMB]], %[[Q_ABS_LIMB]] : i32
  // CHECK:     %[[ABS_LIMB_IDX:.*]] = arith.index_cast %[[ABS_LIMB]] : i32 to index
  // CHECK:     %[[PRIMES:.*]] = memref.get_global @__cheddar_primes : memref<4xi32>
  // CHECK:     %[[Q:.*]] = memref.load %[[PRIMES]][%[[ABS_LIMB_IDX]]] : memref<4xi32>
  // CHECK:     %[[COEFF_IDX:.*]] = arith.index_cast %[[COEFF]] : i32 to index
  // CHECK:     %[[LOCAL_LIMB_IDX:.*]] = arith.index_cast %[[LOCAL_LIMB]] : i32 to index
  // CHECK:     %[[LHS:.*]] = memref.load %[[ARG1]][%[[LOCAL_LIMB_IDX]], %[[COEFF_IDX]]]
  // CHECK:     %[[RHS:.*]] = memref.load %[[ARG2]][%[[LOCAL_LIMB_IDX]], %[[COEFF_IDX]]]
  // CHECK:     %[[SUM:.*]] = arith.addi %[[LHS]], %[[RHS]] : i32
  // CHECK:     %[[IS_GE:.*]] = arith.cmpi uge, %[[SUM]], %[[Q]] : i32
  // CHECK:     %[[SUM_SUB_Q:.*]] = arith.subi %[[SUM]], %[[Q]] : i32
  // CHECK:     %[[RES:.*]] = arith.select %[[IS_GE]], %[[SUM_SUB_Q]], %[[SUM]] : i32
  // CHECK:     memref.store %[[RES]], %[[ARG0]][%[[LOCAL_LIMB_IDX]], %[[COEFF_IDX]]]
  // CHECK:     gpu.return
  // CHECK:   }
  // CHECK:   gpu.func @cheddar_gpu_sub_w32_deg65536
  // CHECK-SAME: (%[[S_ARG0:.*]]: memref<?x65536xi32>, %[[S_ARG1:.*]]: memref<?x65536xi32>, %[[S_ARG2:.*]]: memref<?x65536xi32>,
  // CHECK-SAME:  %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) kernel {
  // CHECK:     %[[S_PRIMES:.*]] = memref.get_global @__cheddar_primes : memref<4xi32>
  // CHECK:     %[[S_Q:.*]] = memref.load %[[S_PRIMES]][%{{.*}}] : memref<4xi32>
  // CHECK:     %[[S_LHS:.*]] = memref.load %[[S_ARG1]]
  // CHECK:     %[[S_RHS:.*]] = memref.load %[[S_ARG2]]
  // CHECK:     %[[DIFF:.*]] = arith.subi %[[S_LHS]], %[[S_RHS]] : i32
  // CHECK:     %[[IS_LT:.*]] = arith.cmpi ult, %[[S_LHS]], %[[S_RHS]] : i32
  // CHECK:     %[[DIFF_ADD_Q:.*]] = arith.addi %[[DIFF]], %[[S_Q]] : i32
  // CHECK:     %[[S_RES:.*]] = arith.select %[[IS_LT]], %[[DIFF_ADD_Q]], %[[DIFF]] : i32
  // CHECK:     memref.store %[[S_RES]], %[[S_ARG0]]
  // CHECK:     gpu.return
  // CHECK:   }
  // CHECK-NOT: gpu.func @cheddar_gpu_add_w32_deg65536
  // CHECK-NOT: gpu.func @cheddar_gpu_sub_w32_deg65536

  // CHECK: func.func @test_add_basic
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[GRID:.*]] = arith.constant 1024 : index
  // CHECK: %[[BLOCK:.*]] = arith.constant 256 : index
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w32_deg65536 blocks in (%[[GRID]], %[[C1]], %[[C1]]) threads in (%[[BLOCK]], %[[C1]], %[[C1]]) args(
  func.func @test_add_basic(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_add_kernel_reuse
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w32_deg65536
  func.func @test_add_kernel_reuse(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_add_with_aux
  // CHECK-DAG: %[[PRIME_OFFSET:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[NUM_Q:.*]] = arith.constant 3 : i32
  // CHECK-DAG: %[[AUX_START:.*]] = arith.constant 3 : i32
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w32_deg65536
  // CHECK-SAME: %[[PRIME_OFFSET]] : i32, %[[NUM_Q]] : i32, %[[AUX_START]] : i32)
  func.func @test_add_with_aux(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval,
      num_aux = 1 : i64
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_sub_basic
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_sub_w32_deg65536
  func.func @test_sub_basic(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.sub %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_sub_kernel_reuse
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_sub_w32_deg65536
  func.func @test_sub_kernel_reuse(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.sub %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_add_and_sub
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w32_deg65536
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_sub_w32_deg65536
  func.func @test_add_and_sub(%out1: memref<4x65536xi32>, %out2: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.add %out1, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<4x65536xi32>
    cheddar_gpu.sub %out2, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<4x65536xi32>
    return
  }
}

// -----

#eval = #cheddar_gpu.form<eval>
#params_64 = #cheddar_gpu.params<word = 64, log_n = 16, ter = [17], main = [97, 193], aux = [257], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>

module attributes {cheddar_gpu.params = #params_64} {

  // CHECK: gpu.module @cheddar_kernels
  // CHECK:   memref.global "private" constant @__cheddar_primes : memref<4xi64> = dense<[17, 97, 193, 257]>
  // CHECK:   gpu.func @cheddar_gpu_add_w64_deg65536
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?x65536xi64>, %[[ARG1:.*]]: memref<?x65536xi64>, %[[ARG2:.*]]: memref<?x65536xi64>,
  // CHECK-SAME:  %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) kernel {
  // CHECK:     %[[PRIMES:.*]] = memref.get_global @__cheddar_primes : memref<4xi64>
  // CHECK:     %[[Q:.*]] = memref.load %[[PRIMES]]
  // CHECK:     %[[LHS:.*]] = memref.load %[[ARG1]]
  // CHECK:     %[[RHS:.*]] = memref.load %[[ARG2]]
  // CHECK:     %[[SUM:.*]] = arith.addi %[[LHS]], %[[RHS]] : i64
  // CHECK:     %[[IS_GE:.*]] = arith.cmpi uge, %[[SUM]], %[[Q]] : i64
  // CHECK:     %[[SUM_SUB_Q:.*]] = arith.subi %[[SUM]], %[[Q]] : i64
  // CHECK:     %[[RES:.*]] = arith.select %[[IS_GE]], %[[SUM_SUB_Q]], %[[SUM]] : i64
  // CHECK:     memref.store %[[RES]], %[[ARG0]]
  // CHECK:     gpu.return
  // CHECK:   }

  // CHECK: func.func @test_add_64bit
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w64_deg65536
  func.func @test_add_64bit(%out: memref<4x65536xi64>, %lhs: memref<4x65536xi64>, %rhs: memref<4x65536xi64>) {
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<4x65536xi64>
    return
  }
}
