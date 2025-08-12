#ifndef LIB_DIALECT_MGMT_TRANSFORMS_UTILS_H_
#define LIB_DIALECT_MGMT_TRANSFORMS_UTILS_H_

#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Client helper functions require mgmt attributes in order for the type
// converter to convert them. The correct mgmt attribute to use is
// determined by the function the helpers are made for, and so
// this method copies the relevant mgmt attributes from
// the original function to the client helper function arg/result attrs
// so they can be propagated through those IRs. Any existing mgmt attributes
// on the client helpers are replaced.
//
// E.g., given the following functions
//
//   func.func @foo(
//      %arg0: !secret.secret<tensor<8xi16>> {mgmt.mgmt = ...})
//         -> (!secret.secret<tensor<8xi16>> {mgmt.mgmt = ...}) {
//     ...
//   }
//   func.func @foo__encrypt__arg0(
//      %arg0: tensor<8xi16>) -> !secret.secret<tensor<8xi16>>
//      attributes {client_enc_func = {func_name = "foo", index = 0 : i64}} {
//     ...
//   }
//   func.func @foo__decrypt__result0(
//      %arg0: !secret.secret<tensor<8xi16>>) -> i16
//       attributes {client_dec_func = {func_name = "foo", index = 0 : i64}} {
//     ...
//   }
//
// The encrypt function needs its result annotated with a mgmt attr that
// matches the mgmt attr of @foo's 0th argument, and the decrypt function's
// argument needs a mgmt attr that matches the mgmt attr of @foo's 0th result.
// Then the mgmt attr needs to be backward propagated in the encryption
// function and forward propagated in the decryption function.
LogicalResult copyMgmtAttrToClientHelpers(Operation* op);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MGMT_TRANSFORMS_UTILS_H_
