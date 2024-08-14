add_subdirectory(Transforms)

add_mlir_dialect_library(MLIRHeirPolynomialNTTRewrites
        Transforms/NTTRewrites.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/Transforms

        DEPENDS
        MLIRPolynomialPassesIncGen
        MLIRPolynomialNTTRewritePassesIncGen

        LINK_LIBS PUBLIC

        MLIRModArithDialect
        MLIRArithDialect
        MLIRIR
        MLIRPass
        MLIRPolynomialDialect
        MLIRTransformUtils
)
