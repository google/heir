add_heir_pass(ConvertIfToSelect)

add_mlir_library(HEIRConvertIfToSelect
    ConvertIfToSelect.cpp

    DEPENDS
    HEIRConvertIfToSelectIncGen

    LINK_LIBS PUBLIC
    HEIRSecretnessAnalysis
    MLIRIR
    MLIRSupport
    MLIRDialect
)
target_link_libraries(HEIRTransforms INTERFACE HEIRConvertIfToSelect)
