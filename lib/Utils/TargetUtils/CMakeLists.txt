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
