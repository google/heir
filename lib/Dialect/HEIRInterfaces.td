include "mlir/IR/OpBase.td"

def LUTOpInterface : OpInterface<"LUTOpInterface"> {
  let description = [{
    This is an example interface definition.
  }];

  let methods = [
    InterfaceMethod<
      "Gets lookup table inputs from most significant bit to least.",
      "std::optional<mlir::ValueRange>", "getLookupTableInputs"
    >,
  ];
}

def RelinearizeOpInterface : OpInterface<"RelinearizeOpInterface"> {
  let description = [{
    This Operation Interface exposes common methods for relinearize operations.
  }];

  let methods = [
    InterfaceMethod<
      "Gets the key basis of the input ciphertext.",
      "SmallVector<int32_t>", "getFromBasis"
    >,
  ];
}

def RotateOpInterface : OpInterface<"RotateOpInterface"> {
  let description = [{
    This Operation Interface exposes common methods for rotate operations.
  }];

  let methods = [
    InterfaceMethod<
      "Gets the rotation offset of the input ciphertext.",
      "int64_t", "getRotationOffset"
    >,
  ];
}
