set(LLVM_TARGET_DEFINITIONS LinalgToTensorExt.td)
mlir_tablegen(LinalgToTensorExt.h.inc -gen-pass-decls -name LinalgToTensorExt)
add_public_tablegen_target(MLIRLinalgToTensorExtTransformsIncGen)

add_mlir_dialect_library(MLIRLinalgToTensorExt
    LinalgToTensorExt.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRLinalgToTensorExtTransformsIncGen

    LINK_LIBS PUBLIC
        MLIRHEIRSecretnessAnalysis
        MLIRTensorExt
        LLVMSupport
        MLIRAffineDialect
        MLIRAnalysis
        MLIRArithDialect
        MLIRIR
        MLIRLinalgDialect
        MLIRPass
        MLIRSupport
        MLIRTensorDialect
        MLIRTransformUtils
)
