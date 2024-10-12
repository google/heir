add_heir_pass(ConvertSecretInsertToStaticInsert)

add_mlir_library(HEIRConvertSecretInsertToStaticInsert
    ConvertSecretInsertToStaticInsert.cpp

    DEPENDS
    HEIRConvertSecretInsertToStaticInsertIncGen

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
target_link_libraries(HEIRTransforms INTERFACE HEIRConvertSecretInsertToStaticInsert)
