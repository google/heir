add_heir_pass(ConvertSecretForToStaticFor)

add_mlir_library(HEIRConvertSecretForToStaticFor
    ConvertSecretForToStaticFor.cpp

    DEPENDS
    HEIRConvertSecretForToStaticForIncGen

    LINK_LIBS PUBLIC
    HEIRSecretnessAnalysis
    LLVMSupport
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSCFDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRDialect
    MLIRTransformUtils
)
target_link_libraries(HEIRTransforms INTERFACE HEIRConvertSecretForToStaticFor)
