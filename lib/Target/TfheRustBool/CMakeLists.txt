
add_mlir_library(HEIRTfheRustBoolEmitter
        TfheRustBoolEmitter.cpp


        LINK_LIBS PUBLIC
        HEIRTfheRustBool
        MLIRIR
        MLIRInferTypeOpInterface
)
target_link_libraries(HEIRTarget INTERFACE HEIRTfheRustBoolEmitter)
