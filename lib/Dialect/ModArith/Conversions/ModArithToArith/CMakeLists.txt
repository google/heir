add_heir_pass(ModArithToArith PATTERNS)

add_mlir_conversion_library(HEIRModArithToArith
    ModArithToArith.cpp

    DEPENDS
    HEIRModArithToArithIncGen

    LINK_LIBS PUBLIC
    HEIRModArith

    LINK_LIBS PUBLIC

    LLVMSupport

    MLIRArithDialect
    MLIRDialect
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTransformUtils
)
