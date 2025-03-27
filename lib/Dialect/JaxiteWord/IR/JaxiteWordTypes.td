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
  string asmName = ?;
}

def JaxiteWord_ModulusList : JaxiteWord_Type<"ModulusList", "modulus_list"> {
  let description = [{
    A list of modulus values.
  }];
  let parameters = (ins ArrayRefParameter<"::mlir::IntegerAttr">:$modulus_list);
  let assemblyFormat = "`<` $modulus_list `>`";
  // jaxiteword.modulus_list<65535, 1152921504606844513, 1152921504606844417>
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

def JaxiteWord_PublicKey : JaxiteWord_Type<"PublicKey", "public_key"> {
  let summary = "The public key required to encrypt plaintext in JaxiteWord.";
  let asmName = "pk";
}

def JaxiteWord_PrivateKey : JaxiteWord_Type<"PrivateKey", "private_key"> {
  let summary = "The private key required to decrypt a ciphertext in JaxiteWord.";
  let asmName = "sk";
}

def JaxiteWord_PublicKeyOrPrivateKey : AnyTypeOf<[JaxiteWord_PublicKey, JaxiteWord_PrivateKey]>;

def JaxiteWord_EvalKey : JaxiteWord_Type<"EvalKey", "eval_key"> {
  let summary = "The evaluation key required to keyswitch/relinearize/rotate/automorphism operation in JaxiteWord.";
  let asmName = "ek";
}

def JaxiteWord_CCParams : JaxiteWord_Type<"CCParams", "cc_params"> {
  let summary = "The CCParams required to create CryptoContext.";
  let asmName = "params";
}

def JaxiteWord_CryptoContext : JaxiteWord_Type<"CryptoContext", "crypto_context"> {
  let summary = "The CryptoContext required to perform homomorphic operations in JaxiteWord.";
  let asmName = "cc";
}


#endif  // LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDTYPES_TD_
