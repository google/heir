add_heir_pass(ForwardInsertToExtract)

add_mlir_library(HEIRForwardInsertToExtract
    ForwardInsertToExtract.cpp

    DEPENDS
    HEIRForwardInsertToExtractIncGen

    LINK_LIBS PUBLIC

    LLVMSupport
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRDialect
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRTransforms
    MLIRTransformUtils
)
target_link_libraries(HEIRTransforms INTERFACE HEIRForwardInsertToExtract)
