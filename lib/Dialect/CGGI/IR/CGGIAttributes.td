#ifndef LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_TD_
#define LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_TD_

include "lib/Dialect/CGGI/IR/CGGIDialect.td"
include "mlir/IR/AttrTypeBase.td"

def CGGI_CGGIParams : AttrDef<CGGI_Dialect, "CGGIParams"> {
  let mnemonic = "cggi_params";

  // TODO(#276): migrate the gadget params
  // to lwe dialect?
  let parameters = (ins
    "::mlir::heir::lwe::RLWEParamsAttr": $rlweParams,
    "unsigned": $bsk_noise_variance,
    "unsigned": $bsk_gadget_base_log,
    "unsigned": $bsk_gadget_num_levels,
    "unsigned": $ksk_noise_variance,
    "unsigned": $ksk_gadget_base_log,
    "unsigned": $ksk_gadget_num_levels
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

#endif  // LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_TD_
