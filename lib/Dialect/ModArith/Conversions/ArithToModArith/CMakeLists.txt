add_heir_pass(ArithToModArith PATTERNS)

add_mlir_conversion_library(HEIRArithToModArith
    ArithToModArith.cpp

    DEPENDS
    HEIRArithToModArithIncGen

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
