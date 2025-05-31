#ifndef LIB_TRANSFORMS_SETPLAINTEXTPARAMS_SETPLAINTEXTPARAMS_TD_
#define LIB_TRANSFORMS_SETPLAINTEXTPARAMS_SETPLAINTEXTPARAMS_TD_

include "mlir/Pass/PassBase.td"

def SetPlaintextParams : Pass<"set-plaintext-params"> {
  let summary = "Set params needed for the plaintext backend";
  let description = [{
  The plaintext backend includes packing and encoding of data types into
  `mod_arith`, which for floating point input types requires tracking the scale
  of the encoding (in the same sense as CKKS).

  This pass sets a module attribute with the scale parameter needed to enable
  the later `populate-scale-plaintext` pass.
  }];
  let options = [
    Option<"logScale", "log-scale", "int",
           /*default=*/"0", "The log2 of the scale to use">,
  ];
}

#endif  // LIB_TRANSFORMS_SETPLAINTEXTPARAMS_SETPLAINTEXTPARAMS_TD_
