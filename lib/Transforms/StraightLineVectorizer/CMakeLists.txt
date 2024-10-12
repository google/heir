add_heir_pass(StraightLineVectorizer)

add_mlir_library(HEIRStraightLineVectorizer
    StraightLineVectorizer.cpp

    DEPENDS
    HEIRStraightLineVectorizerIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
)
target_link_libraries(HEIRTransforms INTERFACE HEIRStraightLineVectorizer)
