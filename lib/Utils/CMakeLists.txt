add_library(HEIRUtils INTERFACE)

add_subdirectory(DRR)
add_subdirectory(Graph)

add_mlir_library(HEIRConversionUtils
    ConversionUtils.cpp

    # DEPENDS
    # HEIRTensorExtOpsIncGen
    # HEIRTensorExtPassesIncGen
    # HEIRTensorExtInsertRotateIncGen
    # HEIRTensorExtCanonicalizationIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
)

add_mlir_library(HEIRTargetUtils
    TargetUtils.cpp

    LINK_LIBS PUBLIC
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)

target_link_libraries(HEIRUtils INTERFACE HEIRTargetUtils)
target_link_libraries(HEIRUtils INTERFACE HEIRConversionUtils)
