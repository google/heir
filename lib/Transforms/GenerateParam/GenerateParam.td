#ifndef LIB_TRANSFORMS_GENERATEPARAM_GENERATEPARAM_TD_
#define LIB_TRANSFORMS_GENERATEPARAM_GENERATEPARAM_TD_

include "mlir/Pass/PassBase.td"

def GenerateParamBGV : Pass<"generate-param-bgv"> {
  let summary = "Generate BGV Scheme Parameter using a given noise model";
  let description = [{
    The pass generates the BGV scheme parameter using a given noise model.

    There are four noise models available:
    - `bgv-noise-by-bound-coeff-average-case` or `bgv-noise-kpz21`
    - `bgv-noise-by-bound-coeff-worst-case`
    - `bgv-noise-by-variance-coeff` or `bgv-noise-mp24`
    - `bgv-noise-mono`

    To use public-key encryption/secret-key encryption in the model, the option
    `usePublicKey` could be set accordingly.

    The first two models are taken from [KPZ21](https://ia.cr/2021/204), and they work by bounding
    the coefficient embedding of the ciphertexts. The difference
    of the two models is expansion factor used for multiplication
    of the coefficients, the first being $2 \sqrt{N}$ ($4 \sqrt{N}$ in some
    special cases) and the second being $N$.

    The third model is taken from [MP24](https://ia.cr/2019/452). It works by tracking the variance
    of the coefficient embedding of the ciphertexts. This gives a more accurate
    noise estimate, but it may give underestimates in some cases. See the paper
    for more details. One possible explanation of the underestimation is [this paper](https://ia.cr/2025/1036).

    The last model is taken from [MMLGA22](https://ia.cr/2022/706). It uses the canonical embedding to
    bound the critical quantity of a ciphertext that defines whether c can be
    decrypted correctly. According to the authors they achieve more accurate and
    better bounds than KPZ21. See the paper for more details.

    This pass relies on the presence of the `mgmt` dialect ops to model
    relinearize/modreduce, and it relies on `mgmt.mgmt` attribute to determine
    the ciphertext level/dimension. These ops and attributes can be added by
    a pass like `--secret-insert-mgmt-bgv`.

    User can provide custom scheme parameters by annotating bgv::SchemeParamAttr
    at the module level.

    (* example filepath=tests/Transforms/generate_param_bgv/doctest.mlir *)
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
  ];

  let options = [
    Option<"model", "model", "std::string",
           /*default=*/"\"bgv-noise-kpz21\"", "Noise model to validate against.">,
    Option<"plaintextModulus", "plaintext-modulus", "int64_t",
           /*default=*/"65537", "Plaintext modulus.">,
    Option<"slotNumber", "slot-number", "int",
           /*default=*/"0", "Minimum number of slots for parameter generation.">,
    Option<"usePublicKey", "use-public-key", "bool", /*default=*/"true",
           "If true, uses a public key for encryption.">,
    Option<"encryptionTechniqueExtended", "encryption-technique-extended", "bool", /*default=*/"false",
           "If true, uses EXTENDED encryption technique for encryption. (See https://ia.cr/2022/915)">,
  ];
}

def GenerateParamBFV : Pass<"generate-param-bfv"> {
  let summary = "Generate BFV Scheme Parameter";
  let description = [{
    The pass generates the BFV scheme parameter using a given noise model.

    There are four noise models available:
    - `bfv-noise-by-bound-coeff-average-case` or `bfv-noise-kpz21`
    - `bfv-noise-by-bound-coeff-worst-case`
    - `bfv-noise-by-variance-coeff` or `bfv-noise-bmcm23`
    - `bfv-noise-canon-emb`

    To use public-key encryption/secret-key encryption in the model, the option
    `usePublicKey` could be set accordingly.

    The first two models are taken from [KPZ21](https://ia.cr/2021/204), and they work by bounding
    the coefficient embedding of the ciphertexts. The difference
    of the two models is expansion factor used for multiplication
    of the coefficients, the first being $2 \sqrt{N}$ ($4 \sqrt{N}$ in some
    special cases) and the second being $N$.

    The third model is taken from [BMCM23](https://ia.cr/2023/600). It works by tracking the variance
    of the coefficient embedding of the ciphertexts. This gives a much tighter
    noise estimate for independent ciphertext input, but may give underestimation
    for dependent ciphertext input. See the paper for more details.
    One possible explanation of the underestimation is [this paper](https://ia.cr/2025/1036)

    The last model is adapted from [MMLGA22](https://ia.cr/2022/706) with mixes from BMCM23 and KPZ21.
    It uses the canonical embedding to bound the critical quantity of a ciphertext
    that defines whether c can be decrypted correctly.

    This pass then generates the moduli chain consisting of primes
    of bits specified by the `mod-bits` field.

    Usually for B/FV `mod-bits` is set to 60. But when machine word size is
    small, users may also want to set it to 57.

    This pass relies on the presence of the `mgmt` dialect ops to model
    relinearize, and it relies on `mgmt.mgmt` attribute to determine
    the ciphertext level/dimension. These ops and attributes can be added by
    a pass like `--secret-insert-mgmt-bgv` and `--annotate-mgmt`.

    User can provide custom scheme parameters by annotating bgv::SchemeParamAttr
    at the module level. Note that we reuse bgv::SchemeParamAttr for BFV.

    (* example filepath=tests/Transforms/generate_param_bfv/doctest.mlir *)
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
  ];

  let options = [
    Option<"model", "model", "std::string",
           /*default=*/"\"bfv-noise-kpz21\"", "Noise model to validate against.">,
    Option<"modBits", "mod-bits", "int",
           /*default=*/"60", "Default number of bits for all prime coefficient modulus"
           "to use for the ciphertext space.">,
    Option<"slotNumber", "slot-number", "int",
           /*default=*/"0", "Minimum number of slots for parameter generation.">,
    Option<"plaintextModulus", "plaintext-modulus", "int64_t",
           /*default=*/"65537", "Plaintext modulus.">,
    Option<"usePublicKey", "use-public-key", "bool", /*default=*/"true",
           "If true, uses a public key for encryption.">,
    Option<"encryptionTechniqueExtended", "encryption-technique-extended", "bool", /*default=*/"false",
           "If true, uses EXTENDED encryption technique for encryption. (See https://ia.cr/2022/915)">,
  ];
}

def GenerateParamCKKS : Pass<"generate-param-ckks"> {
  let summary = "Generate CKKS Scheme Parameter";
  let description = [{
    The pass generates the CKKS scheme parameter.

    The pass asks the user to provide the number of bits for the first modulus
    and scaling modulus. The default values are 55 and 45, respectively.
    Then the pass generates the moduli chain using the provided values.

    This pass relies on the presence of the `mgmt` dialect ops to model
    relinearize/modreduce, and it relies on `mgmt.mgmt` attribute to determine
    the ciphertext level/dimension. These ops and attributes can be added by
    a pass like `--secret-insert-mgmt-<scheme>` and `--annotate-mgmt`.

    User can provide custom scheme parameters by annotating bgv::SchemeParamAttr
    at the module level.

    (* example filepath=tests/Transforms/generate_param_ckks/doctest.mlir *)
  }];

  let dependentDialects = [
    "mlir::heir::ckks::CKKSDialect",
  ];

  let options = [
    Option<"slotNumber", "slot-number", "int",
           /*default=*/"0", "Minimum number of slots for parameter generation.">,
    Option<"firstModBits", "first-mod-bits", "int",
           /*default=*/"55", "Default number of bits of the first prime "
           "coefficient modulus to use for the ciphertext space.">,
    Option<"scalingModBits", "scaling-mod-bits", "int",
           /*default=*/"45", "Default number of bits of the scaling prime "
           "coefficient modulus to use for the ciphertext space.">,
    Option<"usePublicKey", "use-public-key", "bool", /*default=*/"true",
           "If true, uses a public key for encryption.">,
    Option<"encryptionTechniqueExtended", "encryption-technique-extended", "bool", /*default=*/"false",
           "If true, uses EXTENDED encryption technique for encryption. (See https://ia.cr/2022/915)">,
    Option<"inputRange", "input-range", "int",
           /*default=*/"1", "The range of the plaintexts for input ciphertexts "
           "for the CKKS scheme; default to [-1, 1]. For other ranges like [-D, D], use D.">,
  ];
}

#endif  // LIB_TRANSFORMS_GENERATEPARAM_GENERATEPARAM_TD_
