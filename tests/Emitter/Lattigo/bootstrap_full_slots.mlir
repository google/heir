// RUN: heir-translate %s --emit-lattigo | FileCheck %s

// A module carrying scheme.requested_slot_count (set by generate-param-ckks
// from the requested ciphertext-degree, here 1024 in a LogN-16 ring whose
// actual slot count is 32768) must NOT turn that hint into a sparse
// bootstrapping LogSlots.

// CHECK: bootstrapping.NewParametersFromLiteral
// CHECK:   LogN: utils.Pointy(16)
// CHECK-NOT: LogSlots
// CHECK: })

!params = !lattigo.ckks.parameter
!bt_params = !lattigo.ckks.bootstrapping_parameter

#paramsLiteral = #lattigo.ckks.parameters_literal<
    logN = 16,
    logQ = [55, 45, 45],
    logP = [61],
    logDefaultScale = 45
>

module attributes {scheme.ckks, scheme.requested_slot_count = 1024 : i64} {
  func.func @make_bt_params() -> !bt_params {
    %params = lattigo.ckks.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !params
    %bt_params = lattigo.ckks.new_bootstrapping_parameters_from_literal %params {btParamsLiteral = #lattigo.ckks.bootstrapping_parameters_literal<logN = 16>} : (!params) -> !bt_params
    return %bt_params : !bt_params
  }
}
