add_heir_pass(ConvertSecretWhileToStaticFor)

add_mlir_library(HEIRConvertSecretWhileToStaticFor
    ConvertSecretWhileToStaticFor.cpp

    DEPENDS
    HEIRConvertSecretWhileToStaticForIncGen

    LINK_LIBS PUBLIC
    HEIRSecretnessAnalysis
    LLVMSupport
    MLIRAnalysis
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSCFDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransformUtils
)
target_link_libraries(HEIRTransforms INTERFACE HEIRConvertSecretWhileToStaticFor)
