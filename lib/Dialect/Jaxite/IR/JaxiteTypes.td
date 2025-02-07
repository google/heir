#ifndef LIB_DIALECT_JAXITE_IR_JAXITETYPES_TD_
#define LIB_DIALECT_JAXITE_IR_JAXITETYPES_TD_

include "JaxiteDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class Jaxite_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Jaxite_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Jaxite_ServerKeySet : Jaxite_Type<"ServerKeySet", "server_key_set"> {
  let summary = "The jaxite server key set required to perform homomorphic operations.";
}

def Jaxite_Params : Jaxite_Type<"Params", "params"> {
  let summary = "The jaxite security params required to perform homomorphic operations.";
}



def Jaxite_PmapLut3Tuple : Jaxite_Type<"PmapLut3Tuple", "pmap_lut3_tuple"> {
  let summary = "A tuple of pmap_lut3 args.";
}

def Jaxite_PmapLut3Tuples : TypeOrValueSemanticsContainer<Jaxite_PmapLut3Tuple, "pmap_lut3_tuples">;


#endif  // LIB_DIALECT_JAXITE_IR_JAXITETYPES_TD_
