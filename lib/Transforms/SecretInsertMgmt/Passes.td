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
      %0 = secret.generic(%arg0, %arg1 : !secret.secret<i16>, !secret.secret<i16>) {
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
      %0 = secret.generic(%arg0, %arg1 : !secret.secret<i16>, !secret.secret<i16>) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 1>}, arg1 = {mgmt.mgmt = #mgmt.mgmt<level = 1>}} {
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
    "mlir::heir::mgmt::MgmtDialect"
  ];

  let options = [
    Option<"afterMul", "after-mul", "bool",
           /*default=*/"false", "Modulus switching after each multiplication (default to false)">,
    Option<"beforeMulIncludeFirstMul", "before-mul-include-first-mul", "bool",
           /*default=*/"false", "Modulus switching before each multiplication, including the first multiplication (default to false)">,
  ];
}

def SecretInsertMgmtBFV : Pass<"secret-insert-mgmt-bfv", "ModuleOp"> {
  let summary = "Place BFV ciphertext management operations";

  let description = [{
    This pass inserts relinearization operation for multiplication, and
    compute the multiplicative depth, or the level information.

    For most cases B/FV is instantiated with no mod reduce so it is not a leveled scheme.
    However, for instantiating B/FV parameters it is often meaningful to know the multiplicative
    depth of the circuit.

    Example of multiplication+addition:
    ```mlir
    func.func @func(%arg0: !secret.secret<i16>, %arg1: !secret.secret<i16>) -> !secret.secret<i16> {
      %0 = secret.generic(%arg0, %arg1 : !secret.secret<i16>, !secret.secret<i16>) {
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
      %0 = secret.generic(%arg0, %arg1 : !secret.secret<i16>, !secret.secret<i16>) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 1>}, arg1 = {mgmt.mgmt = #mgmt.mgmt<level = 1>}} {
      ^body(%input0: i16, %input1: i16):
        %1 = arith.muli %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : i16
        %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
        %3 = arith.addi %2, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
        secret.yield %3 : i16
      } -> !secret.secret<i16>
      return %0 : !secret.secret<i16>
    }
    ```
  }];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::mgmt::MgmtDialect"
  ];
}

def SecretInsertMgmtCKKS : Pass<"secret-insert-mgmt-ckks", "ModuleOp"> {
  let summary = "Place CKKS ciphertext management operations";

  let description = [{
    Check the description of secret-insert-mgmt-bgv. This pass
    implements similar strategy, where mgmt.modreduce stands for
    ckks.rescale here.

    For bootstrap insertion policy, currently a greedy policy is used
    where when all levels are consumed then a bootstrap is inserted.

    The max level available after bootstrap is controlled by the option
    `bootstrap-waterline`.

    Number of bootstrap consumed level is not shown here, which is
    handled by further lowering.
    TODO(#1207): handle it here so parameter selection can depend on it.
    TODO(#1207): with this info we can encrypt at max level (with bootstrap consumed level).
  }];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::mgmt::MgmtDialect"
  ];

  let options = [
    Option<"afterMul", "after-mul", "bool",
           /*default=*/"false", "Modulus switching after each multiplication (default to false)">,
    Option<"beforeMulIncludeFirstMul", "before-mul-include-first-mul", "bool",
           /*default=*/"false", "Modulus switching before each multiplication, including the first multiplication (default to false)">,
    Option<"slotNumber", "slot-number", "int",
           /*default=*/"1024", "Default number of slots use for ciphertext space.">,
    Option<"bootstrapWaterline", "bootstrap-waterline", "int",
           /*default=*/"10", "Waterline for insert bootstrap op">,
  ];
}

def SecretInsertMgmtPlaintext : Pass<"secret-insert-mgmt-plaintext", "ModuleOp"> {
  let summary = "Place plaintext scale management operations";

  let description = [{
    This is the plaintext analogue of `insert-mgmt-ckks`, where the only thing
    that needs to be managed is the plaintext scale. I.e., we need to apply
    `mod_reduce`/rescaling operations after multiplications, and ensure
    cleartexts that are encoded for "ciphertext-plaintext" operations receive
    the correct scale via `mgmt.init`.
  }];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::mgmt::MgmtDialect"
  ];
}

#endif  // LIB_TRANSFORMS_SECRETINSERTMGMT_PASSES_TD_
