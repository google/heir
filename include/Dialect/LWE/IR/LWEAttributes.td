#ifndef HEIR_INCLUDE_DIALECT_LWE_IR_LWEATTRIBUTES_TD_
#define HEIR_INCLUDE_DIALECT_LWE_IR_LWEATTRIBUTES_TD_

include "LWEDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

class LWE_Attr<string attrName, string attrMnemonic>
    : AttrDef<LWE_Dialect, attrName> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

// TODO: should this live in a dedicated noise analysis dialect so it can be
// easily reused?
def GaussianAttr : LWE_Attr<"Gaussian", "gaussian"> {
  let summary = "An attribute describing a Gaussian distribution over integers.";
  let description = [{
    This attribute describes a symmetric Gaussian distribution used in LWE
    for generating random noise that hides messages.

    The distribution is parameterized by a bit width $B$, representing the
    domain of the distribution, the unsigned integers
    $\mathbb{Z}/B\mathbb{Z}$, and an integer standard deviation. The
    distribution is centered at zero.

    Examples:

    ```
    #lwe_noise = #lwe.gaussian<domain_bitwidth=32, stdev=16384>
    ```
  }];

  let parameters = (ins
    "unsigned":$domain_bitwidth,
    "unsigned":$stdev
  );
}

def LWEEncodingSchemeAttr : LWE_Attr<"LWEEncodingScheme", "encoding_scheme"> {
  let summary = "An attribute describing how cleartexts are encoded into LWE plaintexts.";
  let description = [{
    This attribute contains the parameters of an encoding scheme for LWE.
    An encoding describes the bit width of the plaintext, the number of
    high-order bits reserved for padding, the number of bits to reserve
    for the message. The number of bits reserved for noise is equal to
    the difference of the plaintext bitwidth and the sum of padding and
    cleartext bitwidths, and is always stored in the least significant
    bits of the plaintext.

    Examples:

    ```
    #lwe_encoding = #lwe.encoding_scheme<
      plaintext_bitwidth=32,
      padding_bitwidth=1,
      cleartext_bitwidth=3>
    ```
  }];
  let parameters = (ins
    "unsigned":$plaintext_bitwidth,
    "unsigned":$padding_bitwidth,
    "unsigned":$cleartext_bitwidth
  );

  // Verifier ensures plaintext is large enough to fit the padding and
  // cleartext
  let genVerifyDecl = 1;
}

#endif  // HEIR_INCLUDE_DIALECT_LWE_IR_LWEATTRIBUTES_TD_
