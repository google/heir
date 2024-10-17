# Since this defines two passes in one *.td file,
# we cannot use the standard helper function.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CGGI)
add_public_tablegen_target(HEIRCGGIPassesIncGen)

add_mlir_library(HEIRBooleanVectorizer
    PARTIAL_SOURCES_INTENDED
    BooleanVectorizer.cpp

    DEPENDS
    HEIRCGGIPassesIncGen

    LINK_LIBS PUBLIC
    HEIRCGGI
    HEIRLWE
    HEIRConversionUtils
    MLIRAnalysis
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRTransforms
)

add_mlir_library(HEIRSetDefaultParameters
    PARTIAL_SOURCES_INTENDED
    SetDefaultParameters.cpp

    DEPENDS
    HEIRCGGIPassesIncGen

    LINK_LIBS PUBLIC
    HEIRCGGI
    HEIRLWE
    HEIRConversionUtils
    MLIRAnalysis
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRTransforms
)
