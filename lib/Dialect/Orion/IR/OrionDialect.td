#ifndef LIB_DIALECT_ORION_IR_ORIONDIALECT_TD_
#define LIB_DIALECT_ORION_IR_ORIONDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Orion_Dialect : Dialect {
  let name = "orion";
  let description = [{
    The `orion` dialect is an entry dialect for the [Orion compiler](https://arxiv.org/abs/2311.03470)
    into the heir ecosystem. It is primarily intended to enable comparisons between Orion and HEIR.

    Because an existing translator was written that converts from Orion to HEIR's CKKS dialect,
    this dialect serves to include the ops not supported by the CKKS dialect, but which are
    expressed as black boxes by Orion.
  }];

  let cppNamespace = "::mlir::heir::orion";
}

#endif  // LIB_DIALECT_ORION_IR_ORIONDIALECT_TD_
