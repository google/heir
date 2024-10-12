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

target_link_libraries(HEIRUtils INTERFACE HEIRConversionUtils)
