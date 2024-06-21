add_subdirectory(IR)
add_subdirectory(Transforms)
add_mlir_dialect_library(MLIRSecret
        IR/SecretDialect.cpp
        IR/SecretOps.cpp
        IR/SecretPatterns.cpp
        Transforms/BufferizableOpInterfaceImpl.cpp
        Transforms/CaptureGenericAmbientScope.cpp
        Transforms/DistributeGeneric.cpp
        Transforms/ExtractGenericBody.cpp
        Transforms/ForgetSecrets.cpp
        Transforms/GenericAbsorbConstants.cpp
        Transforms/GenericAbsorbDealloc.cpp
        Transforms/MergeAdjacentGenerics.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRSecretIncGen
        MLIRSecretTypesIncGen
        MLIRSecretOpsIncGen
        MLIRSecretPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
