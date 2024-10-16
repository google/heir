set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Secret)
add_public_tablegen_target(HEIRSecretPassesIncGen)

# Is is useful to have individual libraries for each pass?
add_mlir_library(HEIRSecretTransforms
    BufferizableOpInterfaceImpl.cpp
    CaptureGenericAmbientScope.cpp
    DistributeGeneric.cpp
    ExtractGenericBody.cpp
    ForgetSecrets.cpp
    GenericAbsorbConstants.cpp
    GenericAbsorbDealloc.cpp
    MergeAdjacentGenerics.cpp

    DEPENDS
    HEIRSecret
    HEIRSecretnessAnalysis

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransformUtils
)
