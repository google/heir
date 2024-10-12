# Since we re-set these properties after saving them to local variables
# we should now be able to get the lists for just HEIR from them
get_property(HEIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(HEIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(HEIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(HEIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)


# The three main tools we generate are very similar in their setup
# Dependencies are currently handled in a pretty coarse-grained way
function(add_heir_executable  TARGET_NAME SOURCES)
        add_llvm_executable(${TARGET_NAME} PARTIAL_SOURCES_INTENDED ${SOURCES})
        llvm_update_compile_flags(${TARGET_NAME})
endfunction()

add_heir_executable(heir-opt heir-opt.cpp)
target_link_libraries(heir-opt PRIVATE
    # Utility Function used to define the main entry function
    MLIRMlirOptMain

    # TODO: use more granualar dependency tracking
    ${HEIR_DIALECT_LIBS}
    ${HEIR_EXTENSION_LIBS}
    ${HEIR_CONVERSION_LIBS}
    # ${HEIR_TRANSLATION_LIBS} # No need for translations

    # Anything using add_mlir_library is not automatically included above

    # For most things, there is a convenience target defined:
    HEIRAnalysis # Anything from lib/Anaylysis folder
    HEIRUtils # Anything from lib/Utils folder
    HEIRSource # Anything from lib/Source folder
    HEIRTransforms # Anything from lib/Transforms folder
    # HEIRTarget # Anything from lib/Target folder

    # That just leaves the dialect-specific transforms:
    # TODO: Create an add_mlir_transform or similar function
    # to avoid the need to manually list all dialect-specific transforms
    HEIRBooleanVectorizer
    HEIRLWETransforms
    HEIROpenfheTransforms
    HEIRPolynomialTransforms
    HEIRSecretTransforms
    HEIRSetDefaultParameters
    HEIRTensorExtTransforms
    HEIRTosaToSecretArith

    # # TODO: Optimize CMake MLIR/LLVM dependencies to decrease tool binary sizes
    ${MLIR_DIALECT_LIBS} # allows registerAllDialects()
    ${MLIR_EXTENSION_LIBS} # allows registerAllExtensions()
    ${MLIR_CONVERSION_LIBS} # allows registerAllPasses()
    # ${MLIR_TRANSFORMS_LIBS} # mostly relevant for heir-translate
)

add_heir_executable(heir-translate heir-translate.cpp)
target_link_libraries(heir-translate PRIVATE
    # Utility Function used to define the main entry function
    MLIRTranslateLib

    # TODO: use more granualar dependency tracking
    ${HEIR_DIALECT_LIBS}
    ${HEIR_EXTENSION_LIBS}
    # ${HEIR_CONVERSION_LIBS}
    ${HEIR_TRANSLATION_LIBS}

    HEIRUtils # Anything from lib/Utils folder
    HEIRSource # Anything from lib/Source folder
    HEIRTarget # Anything from lib/Target folder

    # # TODO: Optimize CMake MLIR/LLVM dependencies to decrease tool binary sizes
    ${MLIR_DIALECT_LIBS} # allows registerAllDialects()
    ${MLIR_EXTENSION_LIBS} # allows registerAllExtensions()
    ${MLIR_TRANSFORMS_LIBS} # mostly relevant for heir-translate
)

add_heir_executable(heir-lsp heir-lsp.cpp)
target_link_libraries(heir-lsp PRIVATE
    # Utility Function used to define the main entry function
    MLIRLspServerLib

    # TODO: use more granualar dependency tracking
    ${HEIR_DIALECT_LIBS}
    ${HEIR_EXTENSION_LIBS}

    HEIRUtils # Anything from lib/Utils folder

    # # TODO: Optimize CMake MLIR/LLVM dependencies to decrease tool binary sizes
    ${MLIR_DIALECT_LIBS} # allows registerAllDialects()
    ${MLIR_EXTENSION_LIBS} # allows registerAllExtensions()
)
