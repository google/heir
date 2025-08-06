
// Implementation of an sbox from
// https://core.ac.uk/download/pdf/36694529.pdf

// In the tensor decompisition, the LSB is put on the zero element.
module {
  // multiply in GF(2^2) using normal basis
  // Implemented via bivariate lookup table. Annotate secret.kernel?
  func.func @g4_mul(%arg0: i2, %arg1: i2) -> i2 {
    %concat = comb.concat %arg0, %arg1 : i2, i2
    %ext_concat = arith.extui %concat : i4 to i8
    %lookupTables = arith.constant dense<[0,4,0,0,0,8,0,0,0,0,0,0,0,0,0,0]> : tensor<16xi8>
    %res = comb.lut %ext_concat, %lookupTables : (i8, tensor<16xi8>) -> i8
    %cast = arith.trunci %res : i8 to i2
    return %cast : i2
  }
}

//   func.func @g4_scl_N(%arg0: i2) -> i2 {
//     //TODO: Implement this function in a single PBS
//     return
//   }

//   // Want the bit reverse of the arg0 element
//   func.func @g4_sq(%arg0: i2) -> i2 {
//     // Func: ab : i2 -> ba : i2
//     // TODO: Want to do this in a single PBS. How to express this in MLIR
//     return
//   }
//   func.func @g16_mul(%arg0: tensor<2xi2>, %arg1: tensor<2xi2>) -> tensor<2xi2> {
//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
//     %a = tensor.extract %arg0[%c1] : tensor<2xi2>
//     %b = tensor.extract %arg0[%c0] : tensor<2xi2>
//     %c = tensor.extract %arg1[%c1] : tensor<2xi2>
//     %d = tensor.extract %arg1[%c0] : tensor<2xi2>
//     %e1 = arith.xori %a, %b : i2
//     %e2 = arith.xori %c, %d : i2
//     %e = func.call @g4_mul(%e1, %e2) : (i2, i2) -> i2
//     %e = func.call @g4_scl_N(%e) : (i2) -> i2
//     %p1 = func.call @g4_mul(%a, %c) : (i2, i2) -> i2
//     %p = arith.xori %p1, %e : i2
//     %p2 = func.call @g4_mul(%b, %d) : (i2, i2) -> i2
//     %q = arith.xori %p2, %e : i2
//     %res = tensor.empty() : tensor<2xi2>
//     %res = tensor.insert %p into %res[%c1] : tensor<2xi2>
//     %res = tensor.insert %q into %res[%c0] : tensor<2xi2>
//     return %res
//   }
//   func.func @g16_sq_scl(%arg0: tensor<2xi2>) -> tensor<2xi2> {
//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
//     %a = tensor.extract %arg0[%c1] : tensor<2xi2>
//     %b = tensor.extract %arg0[%c0] : tensor<2xi2>
//     %p1 = arith.xori %a, %b : i2
//     %p = func.call @g4_sq(%p1) : (i2) -> i2
//     %q1 = func.call @g4_sq(%b) : (i2) -> i2
//     %q = func.call @g4_scl_N(%q1) : (i2) -> i2
//     %res = tensor.empty() : tensor<2xi2>
//     %res = tensor.insert %p into %res[%c1] : tensor<2xi2>
//     %res = tensor.insert %q into %res[%c0] : tensor<2xi2>
//     return %res
//   }
//   func.func @g16_inc(%arg0: tensor<2xi2>) -> tensor<2xi2> {
//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
//     %a = tensor.extract %arg0[%c1] : tensor<2xi2>
//     %b = tensor.extract %arg0[%c0] : tensor<2xi2>
//     %c1 = arith.xori %a, %b : i2
//     %c2 = func.call @g4_sq(%c1) : (i2) -> i2
//     %c = func.call @g4_scl_N(%c2) : (i2) -> i2
//     %d = func.call @g4_mul(%a, %b) : (i2, i2) -> i2
//     %e1 = arith.xori %c,%d : i2
//     %e = func.call @g4_sq(%e1) : (i2) -> i2
//     %p = func.call @g4_mul(%e, %b) : (i2, i2) -> i2
//     %q = func.call @g4_mul(%e, %a) : (i2, i2) -> i2
//     %res = tensor.empty() : tensor<2xi2>
//     %res = tensor.insert %p into %res[%c1] : tensor<2xi2>
//     %res = tensor.insert %q into %res[%c0] : tensor<2xi2>
//     return %res
//   }
//   // Invert an element of GF(2^8) using normal basis
//   func.func @g256_inv(%arg0: tensor<4xi2>, %arg1: tensor<4xi2>) -> tensor<4xi2> {
//     // Split arg0 into a top 4 bits and b bottom 4 bits.
//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
//     %ab = secret.cast %arg0 : !secret.secret<i8> to !secret.secret<tensor<4xi2>>
//     %res = secret.generic(%ab : !secret.secret<tensor<4xi2>>) {
//       ^body(%in: tensor<4xi2>):
//       %a = tensor.extract_slice %in[%c0] : tensor<4xi2> to tensor<2xi2>
//       %b = tensor.extract_slice %in[%c1] : tensor<4xi2> to tensor<2xi2>
//       %a_x_b = arith.xori %a %b : i4
//       %c = func.call @g16_sq_scl %a_x_b : (tensor<2xi2>) -> tensor<2xi2>
//       %d = func.call @g16_mul %a, %b : (tensor<2xi2>, tensor<2xi2>) -> tensor<2xi2>
//       %c_x_d = arith.xori %c %d : i4
//       %e = func.call @g16_inv %c_x_d : (i4) -> i4
//       %p = func.call @g16_mul %e, %b : (i4, i4) -> i4
//       %q = func.call @g16_mul %e, %a : (i4, i4) -> i4
//       %res = tensor.concat %p, %q : tensor<2xi8>
//     } -> !secret.secret<tensor<2xi8>>
//     %ret = secret.cast %res : tensor<2xi8> to i8
//     return %ret
//   }
//   func.func @g256_newbasis(%arg0: tensor<4xi2>, %arg1: tensor<8x4xi2>) -> tensor<4xi2> {

//     // So:
//     // Need to unroll the for loop into two calculation per loop.
//     // Parse an x element for both bits inside each loops
//     // Calculate each loop two xor of 8b and do a CMUX on the outcome XOR
//   }


//   func.func @single_sbox(%arg0: i8 {secret.secret}) -> i8 {
//     return
//   }
//   func.func @sbox(%arg0: tensor<16xi8> {secret.secret}) -> tensor<16xi8> {
//     return
//   }
// }
