#ifndef LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_TD_
#define LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_TD_

include "lib/Dialect/Mgmt/IR/MgmtDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

class Mgmt_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Mgmt_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// Custom APInt parameter with safe comparator for scale values
// Note: APIntParameter defines the C++ storage type and uses the default
// FieldParser<APInt> for parsing integer values from text
def ScaleAPIntParameter : APIntParameter<"scale value">;

def Mgmt_MgmtAttr : Mgmt_Attr<"Mgmt", "mgmt"> {
  let summary = "Container attribute for all mgmt parameter";
  let description = [{
    This attribute is used to store all mgmt parameters.

    The attribute is a struct with the following fields:
      - `level` : the level of the ciphertext, from L to 0
      - `dimension` : the dimension of the ciphertext, defaults to 2
      - `scale` : the scale of the ciphertext (high-precision APInt), defaults to 0

    Internally, this attribute is used by secret-to-<scheme>
    for determining the level and dimension of the ciphertext.

    It should be populated by --secret-with-mgmt-<scheme> before
    going through the secret-to-<scheme> pass.

    Example:
    ```
    #mgmt = #mgmt.mgmt<level = 1> // dimension defaults to 2, scale defaults to 0
    #mgmt1 = #mgmt.mgmt<level = 1, dimension = 3>
    %0 = secret.generic(%arg0, %arg1 : !secret.secret<i16>) attrs = {mgmt.mgmt = #mgmt} {
       ...
    } -> !secret.secret<i16>
    ```
  }];

  let parameters = (ins
    "int": $level,
    DefaultValuedParameter<"int", "2">:$dimension,
    ScaleAPIntParameter:$scale
  );
  let builders = [
    AttrBuilder<(ins "int":$level, "int":$dimension), [{
      return $_get(
        $_ctxt,
        level,
        dimension,
        ::llvm::APInt(64, 0));
    }]>,
    AttrBuilder<(ins "int":$level, "int":$dimension, "int64_t":$scale), [{
      return $_get(
        $_ctxt,
        level,
        dimension,
        ::llvm::APInt(64, scale));
    }]>,
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

def Mgmt_OpenfheParamsAttr : Mgmt_Attr<"OpenfheParams", "openfhe_params"> {
  let summary = "Container attribute for some OpenFHE-specific parameters";
  let description = [{
    This attribute is used to store some OpenFHE-specific parameters.

    The attribute is a struct with the following fields:
      - `evalAddCount` : param for OpenFHE SetEvalAddCount
      - `keySwitchCount` : param for OpenFHE SetKeySwitchCount

    When this attribute presents, the lowering of openfhe pass
    will use these parameters to set the corresponding OpenFHE
    parameters.

    It should be populated by --secret-with-mgmt-bgv before
    going through the secret-to-bgv bgv-to-openfhe pass.

    Example:
    ```
    #openfhe_params = #mgmt.openfhe_params<evalAddCount = 1, keySwitchCount = 1>
    ```
  }];

  let parameters = (ins
    "int": $evalAddCount,
    "int": $keySwitchCount
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif  // LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_TD_
