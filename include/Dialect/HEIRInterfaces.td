include "mlir/IR/OpBase.td"

def LUTOpInterface : OpInterface<"LUTOpInterface"> {
  let description = [{
    This is an example interface definition.
  }];

  let methods = [
    InterfaceMethod<
      "Gets lookup table inputs from most significant bit to least.",
      "mlir::ValueRange", "getLookupTableInputs"
    >,
  ];
}
