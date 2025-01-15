
add_mlir_library(HEIRTfheRusHLEmitter
        TfheRusHLEmitter.cpp


        LINK_LIBS PUBLIC
        HEIRTfheRustBool
        MLIRIR
        MLIRInferTypeOpInterface
)
target_link_libraries(HEIRTarget INTERFACE HEIRTfheRusHLEmitter)
