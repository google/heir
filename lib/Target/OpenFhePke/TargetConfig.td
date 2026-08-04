include "lib/Target/CompilationTarget/HEIRTarget.td"

def OpenFHE : CompilationTarget<"openfhe"> {
  // The level budget required here depends on (a) the level budget allowed for
  // encoding and decoding and (b) the secret method (dense/sparse ternary).
  //
  // We can get them dynamically from OpenFHE using FHECKKSRNS::GetBootstrapDepth
  //
  //   UNIFORM_TERNARY with encodeDecodeBudget {1, 1}: 16
  //   SPARSE_TERNARY with encodeDecodeBudget {1, 1}: 12
  //   SPARSE_ENCAPSULATED with encodeDecodeBudget {1, 1}: 12
  //
  // However, Openfhe suggests a selection of {4, 4}, which changes the above
  // to 22 and 18 levels. So we pick the most conservative value below, for
  // UNIFORM_TERNARY with {4, 4}.
  //
  // Cf. https://github.com/google/heir/issues/1207#issuecomment-3678602453
  let bootstrapLevelsConsumed = 22;
}
