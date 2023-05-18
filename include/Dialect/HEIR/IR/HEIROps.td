#ifndef THIRD_PARTY_HEIR_INCLUDE_DIALECT_HEIR_IR_HEIROPS_TD_
#define THIRD_PARTY_HEIR_INCLUDE_DIALECT_HEIR_IR_HEIROPS_TD_

include "HEIRDialect.td"

//===----------------------------------------------------------------------===//
// HEIR op definitions
//===----------------------------------------------------------------------===//

// Base class for the operations in this dialect.
class HEIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<HEIR_Dialect, mnemonic, traits> {

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`  `(` type(operands) `)` `->` type(results)
  }];
}


// An addition operation between two ciphertext values.
def HEIR_AddOp : HEIR_Op<"add"> {
  let summary = "Addition operation between Ciphertexts";

  let arguments = (ins
    Variadic<HEIR_Ciphertext>:$x
  );

  let results = (outs
    HEIR_Ciphertext:$output
  );
}

#endif  // THIRD_PARTY_HEIR_INCLUDE_DIALECT_HEIR_IR_HEIROPS_TD_
