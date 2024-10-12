add_heir_pass(ConvertSecretExtractToStaticExtract)

add_mlir_library(HEIRConvertSecretExtractToStaticExtract
    ConvertSecretExtractToStaticExtract.cpp

    DEPENDS
    HEIRConvertSecretExtractToStaticExtractIncGen

    LINK_LIBS PUBLIC
    HEIRSecretnessAnalysis
    LLVMSupport
    MLIRAffineDialect
    MLIRAnalysis
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSCFDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransformUtils
)
target_link_libraries(HEIRTransforms INTERFACE HEIRConvertSecretExtractToStaticExtract)
