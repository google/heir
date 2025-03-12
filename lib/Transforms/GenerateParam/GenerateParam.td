#ifndef LIB_TRANSFORMS_GENERATEPARAM_GENERATEPARAM_TD_
#define LIB_TRANSFORMS_GENERATEPARAM_GENERATEPARAM_TD_

include "mlir/Pass/PassBase.td"

def GenerateParamBGV : Pass<"generate-param-bgv"> {
  let summary = "Generate BGV Scheme Parameter using a given noise model";
  let description = [{
    The pass generates the BGV scheme parameter using a given noise model.

    There are four noise models available:
    - `bgv-noise-by-bound-coeff-average-case{-pk,-sk}`
    - `bgv-noise-by-bound-coeff-worst-case{-pk,-sk}`
    - `bgv-noise-by-variance-coeff{-pk,-sk}`
    - `bgv-noise-mono{-pk,-sk}`

    The `-pk`/`-sk` suffixes assume the input ciphertexts are
    encrypted using the public/secret key.

    The first two models are taken from KPZ21, and they work by bounding
    the coefficient embedding of the ciphertexts. The difference
    of the two models is expansion factor used for multiplication
    of the coefficients, the first being $2 \sqrt{N}$ and the second
    being $N$.

    The third model is taken from MP24. It works by tracking the variance
    of the coefficient embedding of the ciphertexts. This gives a more accurate
    noise estimate, but it may give underestimates in some cases. See the paper
    for more details.

    The last model is taken from MMLGA22. It uses the canonical embedding to
    bound the critical quantity of a ciphertext that defines whether c can be
    decrypted correctly. According to the authors they achieve more accurate and
    better bounds than KPZ21. See the paper for more details.

    This pass relies on the presence of the `mgmt` dialect ops to model
    relinearize/modreduce, and it relies on `mgmt.mgmt` attribute to determine
    the ciphertext level/dimension. These ops and attributes can be added by
    a pass like `--secret-insert-mgmt-bgv` and `--annotate-mgmt`.

    User can provide custom scheme parameters by annotating bgv::SchemeParamAttr
    at the module level.

    Example:

    ```mlir
    module {
      func.func @add(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
        %0 = secret.generic ins(%arg0 : !secret.secret<i16>) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
        ^body(%input0: i16):
          %1 = arith.addi %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
          secret.yield %1 : i16
        } -> !secret.secret<i16>
        return %0 : !secret.secret<i16>
      }
    }
    ```

    After applying the pass, the module will be updated with the scheme parameters:

    ```mlir
    module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [4294991873], P = [4295049217], plaintextModulus = 65537>} {
      func.func @add(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
        // same as above
      }
    }
    ```
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
  ];

  let options = [
    Option<"model", "model", "std::string",
           /*default=*/"\"bgv-noise-by-bound-coeff-average-case-pk\"", "Noise model to validate against.">,
    Option<"plaintextModulus", "plaintext-modulus", "int64_t",
           /*default=*/"65537", "Plaintext modulus.">,
    Option<"slotNumber", "slot-number", "int",
           /*default=*/"0", "Minimum number of slots for parameter generation.">,
  ];
}

def GenerateParamBFV : Pass<"generate-param-bfv"> {
  let summary = "Generate BFV Scheme Parameter";
  let description = [{
    The pass generates the BFV scheme parameter using a given noise model.

    There are three noise models available:
    - `bfv-noise-by-bound-coeff-average-case{-pk,-sk}`
    - `bfv-noise-by-bound-coeff-worst-case{-pk,-sk}`

    The `-pk`/`-sk` suffixes assume the input ciphertexts are
    encrypted using the public/secret key.

    The first two models are taken from KPZ21, and they work by bounding
    the coefficient embedding of the ciphertexts. The difference
    of the two models is expansion factor used for multiplication
    of the coefficients, the first being $2 \sqrt{N}$ and the second
    being $N$.

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

    Example:

    ```mlir
    module {
      func.func @add(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
        %0 = secret.generic ins(%arg0 : !secret.secret<i16>) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
        ^body(%input0: i16):
          %1 = arith.addi %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
          secret.yield %1 : i16
        } -> !secret.secret<i16>
        return %0 : !secret.secret<i16>
      }
    }
    ```

    After applying the pass, the module will be updated with the scheme parameters:

    ```mlir
    module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [1152921504606994433], P = [1152921504607191041], plaintextModulus = 65537>} {
      func.func @add(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
        // same as above
      }
    }
    ```
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
  ];

  let options = [
    Option<"model", "model", "std::string",
           /*default=*/"\"bfv-noise-by-bound-coeff-average-case-pk\"", "Noise model to validate against.">,
    Option<"modBits", "mod-bits", "int",
           /*default=*/"60", "Default number of bits for all prime coefficient modulus"
           "to use for the ciphertext space.">,
    Option<"slotNumber", "slot-number", "int",
           /*default=*/"0", "Minimum number of slots for parameter generation.">,
    Option<"plaintextModulus", "plaintext-modulus", "int64_t",
           /*default=*/"65537", "Plaintext modulus.">,
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

    Example:

    ```mlir
    module {
      func.func @add(%arg0: !secret.secret<f16>) -> !secret.secret<f16> {
        %0 = secret.generic ins(%arg0 : !secret.secret<f16>) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
        ^body(%input0: f16):
          %1 = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f16
          secret.yield %1 : f16
        } -> !secret.secret<f16>
        return %0 : !secret.secret<f16>
      }
    }
    ```

    After applying the pass, the module will be updated with the scheme parameters:

    ```mlir
    module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953], P = [36028797019488257], logDefaultScale = 45>} {
      func.func @add(%arg0: !secret.secret<f16>) -> !secret.secret<f16> {
        // same as above
      }
    }
    ```
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
  ];
}

#endif  // LIB_TRANSFORMS_GENERATEPARAM_GENERATEPARAM_TD_
