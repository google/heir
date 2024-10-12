
add_mlir_library(HEIRTfheRustTarget
        TfheRustEmitter.cpp
        Utils.cpp

        LINK_LIBS PUBLIC
        HEIRTfheRust
        MLIRIR
        MLIRInferTypeOpInterface
)
target_link_libraries(HEIRTarget INTERFACE HEIRTfheRustTarget)
