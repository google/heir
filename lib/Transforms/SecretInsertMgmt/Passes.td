#ifndef LIB_TRANSFORMS_SECRETINSERTMGMT_PASSES_TD_
#define LIB_TRANSFORMS_SECRETINSERTMGMT_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SecretInsertMgmtBGV : Pass<"secret-insert-mgmt-bgv", "ModuleOp"> {
  let summary = "Place BGV ciphertext management operations";

  let description = [{
    This pass implements the following placement strategy:

    For relinearize, after every homomorphic ciphertext-ciphertext
    multiplication, a mgmt.relinearize is placed after the operation. This is
    done to ensure that the ciphertext keeps _linear_.

    For modulus switching, it is inserted right before a homomorphic
    multiplication, including ciphertext-plaintext ones. There is an option
    `include-first` controlling whether to switch modulus before the first
    multiplication.

    User can check the FLEXIBLEAUTOEXT and FLEXIBLEAUTO mode in OpenFHE as a
    reference. To know more technical difference about them, user can refer to
    the paper ["Revisiting homomorphic encryption schemes for finite firelds"](https://ia.cr/2021/204).

    Then, for level-mismatching binary operations like addition and subtraction,
    additional modulus switch is placed for the operand until it reaches the
    same level.

    This is different from crosslevel operation handling in other
    implementations like using modulus switching and level drop together. The
    reason we only use modulus switching is for simplicity for now. Further
    optimization on this pass could implement such a strategy.

    Before yield the final result, a modulus switching is placed if it is a
    result of multiplication or derived value of a multiplication.

    Also, it annotates the mgmt.mgmt attribute for each operation, which
    includes the level and dimension information of a ciphertext. This
    information is subsequently used by the secret-to-bgv pass to properly
    lower to corresponding RNS Type.

    (* example filepath=tests/Transforms/secret_insert_mgmt_bgv/doctest.mlir *)
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
    Option<"levelBudget", "level-budget", "int", /*default=*/"0",
           "An optional maximum level budget for the pipeline to assume">,
  ];
}

def SecretInsertMgmtBFV : Pass<"secret-insert-mgmt-bfv", "ModuleOp"> {
  let summary = "Place BFV ciphertext management operations";

  let description = [{
    This pass inserts relinearization operation for multiplication, and
    compute the multiplicative depth, or the level information.

    For most cases B/FV is instantiated with no mod reduce so it is not a
    leveled scheme. However, for instantiating B/FV parameters it is often
    meaningful to know the multiplicative depth of the circuit.

    (* example filepath=tests/Transforms/secret_insert_mgmt_bfv/doctest.mlir *)
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
    Option<"levelBudget", "level-budget", "int", /*default=*/"0",
           "An optional maximum level budget for the pipeline to assume">,
  ];
}

#endif  // LIB_TRANSFORMS_SECRETINSERTMGMT_PASSES_TD_
