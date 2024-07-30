set(LLVM_TARGET_DEFINITIONS CombToCGGI.td)
mlir_tablegen(CombToCGGI.h.inc -gen-pass-decls -name CombToCGGI)
add_public_tablegen_target(MLIRCombtTransformsIncGen)

add_mlir_dialect_library(MLIRCombTransforms
    CombToCGGI.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRCombtTransformsIncGen

    LINK_LIBS PUBLIC

    MLIRHEIRUtils
    MLIRCGGI
    MLIRComb
    MLIRLWE
    MLIRSecret

    LLVMSupport
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRDialectUtils
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSupport
    MLIRTransformUtils
)
