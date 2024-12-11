#ifndef LIB_TRANSFORMS_SECRETINSERTMGMT_PASSES_TD_
#define LIB_TRANSFORMS_SECRETINSERTMGMT_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SecretInsertMgmtBGV : Pass<"secret-insert-mgmt-bgv", "ModuleOp"> {
  let summary = "Place BGV ciphertext management operations";

  let description = [{
    This pass implements the following placement strategy:

    For relinearize, after every homomorphic ciphertext-ciphertext multiplication,
    a mgmt.relinearize is placed after the operation. This is done to ensure that
    the ciphertext keeps _linear_.

    For modulus switching, it is inserted right before a homomorphic multiplication,
    including ciphertext-plaintext ones. There is an option `include-first` controlling
    whether to switch modulus before the first multiplication.

    User can check the FLEXIBLEAUTOEXT and FLEXIBLEAUTO mode in OpenFHE as a reference.
    To know more technical difference about them, user can refer to the paper
    ["Revisiting homomorphic encryption schemes for finite firelds"](https://ia.cr/2021/204).

    Then, for level-mismatching binary operations like addition and subtraction,
    additional modulus switch is placed for the operand until it reaches the same level.

    This is different from crosslevel operation handling in other implementations like using
    modulus switching and level drop together. The reason we only use modulus switching is
    for simplicity for now. Further optimization on this pass could implement such a strategy.

    Before yield the final result, a modulus switching is placed if it is a result
    of multiplication or derived value of a multiplication.

    Also, it annotates the mgmt.mgmt attribute for each operation, which
    includes the level and dimension information of a ciphertext. This information
    is subsequently used by the secret-to-bgv pass to properly lower to corresponding
    RNS Type.

    Example of multiplication+addition:
    ```mlir
    func.func @func(%arg0: !secret.secret<i16>, %arg1: !secret.secret<i16>) -> !secret.secret<i16> {
      %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<i16>, !secret.secret<i16>) {
      ^bb0(%arg2: i16, %arg3: i16):
        %1 = arith.muli %arg2, %arg3 : i16
        %2 = arith.addi %1, %arg3 : i16
        secret.yield %2 : i16
      } -> !secret.secret<i16>
      return %0 : !secret.secret<i16>
    }
    ```

    which get transformed to:
    ```mlir
    func.func @func(%arg0: !secret.secret<i16>, %arg1: !secret.secret<i16>) -> !secret.secret<i16> {
      %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<i16>, !secret.secret<i16>) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 1>}, arg1 = {mgmt.mgmt = #mgmt.mgmt<level = 1>}} {
      ^bb0(%arg2: i16, %arg3: i16):
        %1 = arith.muli %arg2, %arg3 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : i16
        %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
        %3 = arith.addi %2, %arg3 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
        %4 = mgmt.modreduce %3 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
        secret.yield %4 : i16
      } -> !secret.secret<i16>
      return %0 : !secret.secret<i16>
    }
    ```
  }];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::mgmt::MgmtDialect",
    "mlir::func::FuncDialect"
  ];

  let options = [
    Option<"includeFirstMul", "include-first-mul", "bool",
           /*default=*/"false", "Modulus switching right before the first multiplication (default to false)">
  ];
}

#endif  // LIB_TRANSFORMS_SECRETINSERTMGMT_PASSES_TD_
