
add_mlir_library(HEIROpenFheRegistration
        OpenFhePkeEmitter.cpp
        OpenFhePkeHeaderEmitter.cpp
        OpenFheUtils.cpp
        OpenFheTranslateRegistration.cpp

        LINK_LIBS PUBLIC
        HEIROpenfhe
        MLIRIR
        MLIRInferTypeOpInterface
)
target_link_libraries(HEIRTarget INTERFACE HEIROpenFheRegistration)
