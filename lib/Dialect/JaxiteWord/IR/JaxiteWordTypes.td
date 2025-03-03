#ifndef LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDTYPES_TD_
#define LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDTYPES_TD_

include "JaxiteWordDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class JaxiteWord_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<JaxiteWord_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
  let assemblyFormat = "`<` struct(params) `>`"; // print out all information in the arguments
}

def JaxiteWord_ModulusList : JaxiteWord_Type<"ModulusList", "modulus_list"> {
  let description = [{
    A list of modulus values.
  }];
  let parameters = (ins ArrayRefParameter<"::mlir::Type">:$modulus_list);
  let assemblyFormat = "`<` $modulus_list `>`";
  // jaxiteword.modulus_list<i32, i32, i32, i32>
}

def JaxiteWord_Ciphertext : JaxiteWord_Type<"Ciphertext", "ciphertext"> {
  let description = [{
    A ciphertext - a three dimensional array.
  }];
  let parameters = (ins "int":$polys,
                    "int":$towers,
                    "int":$degrees);
  let assemblyFormat = "`<` $polys `,` $towers `,` $degrees `>`";
  // jaxiteword.ciphertext<3, 4, 5>
}

#endif  // LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDTYPES_TD_
