
// Implementation of an sbox from
// https://core.ac.uk/download/pdf/36694529.pdf

// For the demo to work, the one-shot-bufferization need to be turned off!

// In the tensor decompisition, the LSB is put on the zero element.
module {
  // multiply in GF(2^2) using normal basis
  // Implemented via bivariate lookup table. Annotate secret.kernel?

  func.func @g4_mul(%arg0: i2, %arg1: i2) -> i2 {
    %ext_arg0 = arith.extui %arg0 : i2 to i8
    %ext_arg1 = arith.extui %arg1 : i2 to i8
    %res = comb.lut %ext_arg0, %ext_arg1 {coefficients = array<i8: 1, 2>, lookupTable = array<i8: 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>} : (i8, i8) -> i8
    %cast = arith.trunci %res : i8 to i2
    return %cast : i2
  }

  func.func @g4_scl_N(%arg0: i2) -> i2 {
    %ext_concat = arith.extui %arg0 : i2 to i8
    %res = comb.lut %ext_concat {coefficients = array<i8: 1>, lookupTable = array<i8: 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>} : (i8) -> i8
    %cast = arith.trunci %res : i8 to i2
    return %cast : i2
  }

  // Want the bit reverse of the arg0 element
  func.func @g4_sq(%arg0: i2) -> i2 {
    %ext_concat = arith.extui %arg0 : i2 to i8
    %res = comb.lut %ext_concat {coefficients = array<i8: 1>, lookupTable = array<i8: 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>} : (i8) -> i8
    %cast = arith.trunci %res : i8 to i2
    return %cast : i2
  }

  func.func @g16_mul(%arg0: tensor<2xi2>, %arg1: tensor<2xi2>) -> tensor<2xi2> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = tensor.extract %arg0[%c1] : tensor<2xi2>
    %b = tensor.extract %arg0[%c0] : tensor<2xi2>
    %c = tensor.extract %arg1[%c1] : tensor<2xi2>
    %d = tensor.extract %arg1[%c0] : tensor<2xi2>
    %e1 = arith.xori %a, %b : i2
    %e2 = arith.xori %c, %d : i2
    %e_old = func.call @g4_mul(%e1, %e2) : (i2, i2) -> i2
    %e = func.call @g4_scl_N(%e_old) : (i2) -> i2
    %p1 = func.call @g4_mul(%a, %c) : (i2, i2) -> i2
    %p = arith.xori %p1, %e : i2
    %p2 = func.call @g4_mul(%b, %d) : (i2, i2) -> i2
    %q = arith.xori %p2, %e : i2
    %res = tensor.empty() : tensor<2xi2>
    %1 = tensor.insert %p into %res[%c1] : tensor<2xi2>
    %2 = tensor.insert %q into %1[%c0] : tensor<2xi2>
    return %2 : tensor<2xi2>
  }
  func.func @g16_sq_scl(%arg0: tensor<2xi2>) -> tensor<2xi2> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = tensor.extract %arg0[%c1] : tensor<2xi2>
    %b = tensor.extract %arg0[%c0] : tensor<2xi2>
    %p1 = arith.xori %a, %b : i2
    %p = func.call @g4_sq(%p1) : (i2) -> i2
    %q1 = func.call @g4_sq(%b) : (i2) -> i2
    %q = func.call @g4_scl_N(%q1) : (i2) -> i2
    %res = tensor.empty() : tensor<2xi2>
    %1 = tensor.insert %p into %res[%c1] : tensor<2xi2>
    %2 = tensor.insert %q into %1[%c0] : tensor<2xi2>
    return %2 : tensor<2xi2>
  }
  func.func @g16_inv(%arg0: tensor<2xi2>) -> tensor<2xi2> {
    %cst0 = arith.constant 0 : index
    %cst1 = arith.constant 1 : index
    %a = tensor.extract %arg0[%cst1] : tensor<2xi2>
    %b = tensor.extract %arg0[%cst0] : tensor<2xi2>
    %c1 = arith.xori %a, %b : i2
    %c2 = func.call @g4_sq(%c1) : (i2) -> i2
    %c = func.call @g4_scl_N(%c2) : (i2) -> i2
    %d = func.call @g4_mul(%a, %b) : (i2, i2) -> i2
    %e1 = arith.xori %c,%d : i2
    %e = func.call @g4_sq(%e1) : (i2) -> i2
    %p = func.call @g4_mul(%e, %b) : (i2, i2) -> i2
    %q = func.call @g4_mul(%e, %a) : (i2, i2) -> i2
    %res = tensor.empty() : tensor<2xi2>
    %1 = tensor.insert %p into %res[%cst1] : tensor<2xi2>
    %2 = tensor.insert %q into %1[%cst0] : tensor<2xi2>
    return %2 : tensor<2xi2>
  }
  // Invert an element of GF(2^8) using normal basis
  // func.func @g256_inv(%arg0: tensor<4xi2>) -> tensor<4xi2> {
  //   // Split arg0 into a top 4 bits and b bottom 4 bits.
  //   %c0 = arith.constant 0 : index
  //   %c1 = arith.constant 1 : index
  //   %arg0_sec = secret.conceal %arg0 : tensor<4xi2> -> !secret.secret<tensor<4xi2>>
  //   %ab = secret.cast %arg0_sec : !secret.secret<tensor<4xi2>> to !secret.secret<tensor<4xi2>>
  //   %res = secret.generic(%ab : !secret.secret<tensor<4xi2>>) {
  //     ^body(%in: tensor<4xi2>):
  //     %a = tensor.extract_slice %in[0] [2] [1] : tensor<4xi2> to tensor<2xi2>
  //     %b = tensor.extract_slice %in[2] [2] [1] : tensor<4xi2> to tensor<2xi2>
  //     %a_x_b = arith.xori %a, %b : tensor<2xi2>
  //     %c = func.call @g16_sq_scl(%a_x_b) : (tensor<2xi2>) -> tensor<2xi2>
  //     %d = func.call @g16_mul(%a, %b) : (tensor<2xi2>, tensor<2xi2>) -> tensor<2xi2>
  //     %c_x_d = arith.xori %c, %d : tensor<2xi2>
  //     %e = func.call @g16_inv(%c_x_d) : (tensor<2xi2>) -> tensor<2xi2>
  //     %p = func.call @g16_mul(%e, %b) : (tensor<2xi2>, tensor<2xi2>) -> tensor<2xi2>
  //     %q = func.call @g16_mul(%e, %a) : (tensor<2xi2>, tensor<2xi2>) -> tensor<2xi2>
  //     %res = tensor.concat dim(0) %p, %q : (tensor<2xi2>, tensor<2xi2>) -> tensor<4xi2>
  //     secret.yield %res : tensor<4xi2>
  //   } -> !secret.secret<tensor<4xi2>>
  //   %ret = secret.reveal %res : !secret.secret<tensor<4xi2>> -> tensor<4xi2>
  //   return %ret : tensor<4xi2>
  // }

  func.func @g256_inv(%arg0: tensor<4xi2>) -> tensor<4xi2> {
    // Split arg0 into a top 4 bits and b bottom 4 bits.
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = tensor.extract_slice %arg0[0] [2] [1] : tensor<4xi2> to tensor<2xi2>
    %b = tensor.extract_slice %arg0[2] [2] [1] : tensor<4xi2> to tensor<2xi2>
    %a_x_b = arith.xori %a, %b : tensor<2xi2>
    %c = func.call @g16_sq_scl(%a_x_b) : (tensor<2xi2>) -> tensor<2xi2>
    %d = func.call @g16_mul(%a, %b) : (tensor<2xi2>, tensor<2xi2>) -> tensor<2xi2>
    %c_x_d = arith.xori %c, %d : tensor<2xi2>
    %e = func.call @g16_inv(%c_x_d) : (tensor<2xi2>) -> tensor<2xi2>
    %p = func.call @g16_mul(%e, %b) : (tensor<2xi2>, tensor<2xi2>) -> tensor<2xi2>
    %q = func.call @g16_mul(%e, %a) : (tensor<2xi2>, tensor<2xi2>) -> tensor<2xi2>
    %ret = tensor.concat dim(0) %p, %q : (tensor<2xi2>, tensor<2xi2>) -> tensor<4xi2>
    return %ret : tensor<4xi2>
  }


// Function to check if a tensor of i2 is equal to a given i2 value
func.func @check_value_x(%arg0: tensor<?xi2>, %arg1: i2) -> i1 {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xi2>

    %c0_i2 = arith.constant 0 : i2

    %empty_tensor = tensor.empty(%dim) : tensor<?xi2>

    %zeros_tensor = linalg.fill ins(%c0_i2 : i2) outs(%empty_tensor : tensor<?xi2>) -> tensor<?xi2>
    %expected_tensor = tensor.insert %arg1 into %zeros_tensor[%c0] : tensor<?xi2>

    %comparison_tensor = arith.cmpi eq, %arg0, %expected_tensor : tensor<?xi2>

    %casted_tensor = arith.extui %comparison_tensor : tensor<?xi1> to tensor<?xi2>
    %initial_sum = arith.constant dense<0> : tensor<i2>

    %sum_tensor = linalg.generic {
    // Define how tensors are accessed.
    // Input (1-D tensor): accessed by the reduction dimension (d0)
    // Output (0-D tensor): has no dimension, so the map is empty
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> ()>
    ],
    // The dimension being reduced is a "reduction" iterator.
    iterator_types = ["reduction"]
  }
    // Operands
    ins(%casted_tensor : tensor<?xi2>)
    outs(%initial_sum : tensor<i2>)
    // Region: The reduction computation is identical.
  {
  ^bb0(%in: i2, %out: i2):
    %sum = arith.addi %in, %out : i2
    linalg.yield %sum : i2
  } -> tensor<i2>


    // Extract the scalar sum from the 0-D result tensor.
    %scalar_sum = tensor.extract %sum_tensor[] : tensor<i2>

    %dim_value = arith.index_castui %dim : index to i2
    // Cmpi operation produces an i1 result
    %result = arith.cmpi eq, %scalar_sum, %dim_value : i2

    // Return the final boolean result.
    return %result : i1
  }



  func.func @g256_newbasis(%arg0: tensor<4xi2>, %arg1: tensor<8x4xi2>) -> tensor<4xi2> {

  //   // So:
  //   // Need to unroll the for loop into two calculation per loop.
  //   // Parse an x element for both bits inside each loops
  //   // Calculate each loop two xor of 8b and do a CMUX on the outcome XOR

      // Define constants for indexing and values.
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index

    %c_1 = arith.constant -1 : index
    %c_2 = arith.constant -2 : index

    %c1_i2 = arith.constant 1 : i2
    %c3_i2 = arith.constant 3 : i2

    %y_initial = arith.constant dense<0> : tensor<4xi2>

    %x_dyn = tensor.cast %arg0 : tensor<4xi2> to tensor<?xi2>

    // Loop 4 times (for 8 bits, unrolled by 2), with a step of 2.
    // The loop carries the accumulator 'y' and the progressively sliced tensor 'x'.
    %y_result, %x_result = affine.for %j = 0 to 4 iter_args(%y_iter = %y_initial, %x_iter = %x_dyn) -> (tensor<4xi2>, tensor<?xi2>) {
      // --- Unrolled Iteration 1 ---

      // The original loop ran with i = 7, 5, 3, 1.
      // Using %j with the formula: i = 7 - (2 * j)
      %two_j = arith.muli %j, %c2 : index
      %i = arith.subi %c7, %two_j : index

      %cond1 = func.call @check_value_x(%x_iter, %c1_i2) : (tensor<?xi2>, i2) -> i1

      %b_i = tensor.extract_slice %arg1[%i, 0] [1, 4] [1, 1] : tensor<8x4xi2> to tensor<4xi2>

      %xor_result = arith.xori %y_iter, %b_i : tensor<4xi2>
      %new_y = arith.select %cond1, %xor_result, %y_iter : tensor<4xi2>

      // --- Unrolled Iteration 2 ---

      %cond2 = func.call @check_value_x(%x_iter, %c3_i2) : (tensor<?xi2>, i2) -> i1

      %i_1 = arith.subi %i, %c1 : index
      %b_i_1 = tensor.extract_slice %arg1[%i_1, 0] [1, 4] [1, 1] : tensor<8x4xi2> to tensor<4xi2>

      %xor_result2 = arith.xori %new_y, %b_i_1 : tensor<4xi2>
      %res_y = arith.select %cond2, %xor_result2, %new_y : tensor<4xi2>

      // Slice the tensor to prepare for the next loop iteration.
      // Remove the LSB (first element) from x.
      %dim_old_2 = tensor.dim %x_iter, %c0 : tensor<?xi2>
      %dim_new_2 = arith.subi %dim_old_2, %c2 : index
      %x_update = tensor.extract_slice %x_iter[2] [%dim_new_2] [1] : tensor<?xi2> to tensor<?xi2>

      // Yield the updated y and x for the next iteration.
      affine.yield %res_y, %x_update : tensor<4xi2>, tensor<?xi2>
    }

    // Return the final computed value of y.
    return %y_result : tensor<4xi2>
  }


  func.func @sbox(%arg0: tensor<4xi2>) -> tensor<4xi2> {

    %A2X = arith.constant dense<[
      [2, 1, 2, 0],  // 0x98
      [3, 3, 0, 3],  // 0xF3
      [3, 3, 0, 2],  // 0xF2
      [1, 0, 2, 0],  // 0x48
      [0, 0, 2, 1],  // 0x09
      [2, 0, 0, 1],  // 0x81
      [2, 2, 2, 1],  // 0xA9
      [3, 3, 3, 3]   // 0xFF
    ]> : tensor<8x4xi2>

    %X2S = arith.constant dense<[
      [1, 1, 2, 0],  // 0x58
      [0, 2, 3, 1],  // 0x2D
      [2, 1, 3, 2],  // 0x9E
      [0, 0, 2, 3],  // 0x0B
      [3, 1, 3, 0],  // 0xDC
      [0, 0, 1, 0],  // 0x04
      [0, 0, 0, 3],  // 0x03
      [0, 2, 1, 0]   // 0x24
    ]> : tensor<8x4xi2>

    %after_basis = func.call @g256_newbasis(%arg0, %A2X) : (tensor<4xi2>, tensor<8x4xi2>) -> tensor<4xi2>
    %inv = func.call @g256_inv(%after_basis) : (tensor<4xi2>) -> tensor<4xi2>
    %out = func.call @g256_newbasis(%inv, %X2S) : (tensor<4xi2>, tensor<8x4xi2>) -> tensor<4xi2>
    %x63 = arith.constant dense<[1, 2, 0, 3]> : tensor<4xi2> // 0x63
    %xor_out = arith.xori %out, %x63 : tensor<4xi2>
    return %xor_out : tensor<4xi2>
  }

  func.func @isbox(%arg0: tensor<4xi2>) -> tensor<4xi2> {

    %X2A = arith.constant dense<[
      [1, 2, 1, 0],  // 0x64
      [1, 3, 2, 0],  // 0x78
      [1, 2, 3, 2],  // 0x6E
      [2, 0, 3, 0],  // 0x8C
      [1, 2, 2, 0],  // 0x68
      [0, 2, 2, 1],  // 0x29
      [3, 1, 3, 2],  // 0xDE
      [1, 2, 0, 0]   // 0x60
    ]> : tensor<8x4xi2>

    %S2X = arith.constant dense<[
      [2, 0, 3, 0],  // 0x8C
      [1, 3, 2, 1],  // 0x79
      [0, 0, 1, 1],  // 0x05
      [3, 2, 2, 3],  // 0xEB
      [0, 1, 0, 2],  // 0x12
      [0, 0, 1, 0],  // 0x04
      [1, 1, 0, 1],  // 0x51
      [1, 1, 0, 3]   // 0x53
    ]> : tensor<8x4xi2>

    %x63 = arith.constant dense<[1, 2, 0, 3]> : tensor<4xi2> // 0x63
    %xor_in = arith.xori %arg0, %x63 : tensor<4xi2>
    %after_basis = func.call @g256_newbasis(%xor_in, %S2X) : (tensor<4xi2>, tensor<8x4xi2>) -> tensor<4xi2>
    %inv = func.call @g256_inv(%after_basis) : (tensor<4xi2>) -> tensor<4xi2>
    %out = func.call @g256_newbasis(%inv, %X2A) : (tensor<4xi2>, tensor<8x4xi2>) -> tensor<4xi2>
    return %out : tensor<4xi2>
  }

}
