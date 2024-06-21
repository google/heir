include(ExternalProject)

# Download RapidJSON
# TODO (#818): automatically extract the commit from the bazel WORKSPACE file
ExternalProject_Add(
    rapidjson
    PREFIX "vendor/rapidjson"
    GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
    GIT_TAG f54b0e47a08782a6131cc3d60f94d038fa6e0a51
    TIMEOUT 10
    CMAKE_ARGS
        -DRAPIDJSON_BUILD_TESTS=OFF
        -DRAPIDJSON_BUILD_DOC=OFF
        -DRAPIDJSON_BUILD_EXAMPLES=OFF
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

# Prepare RapidJSON (RapidJSON is a header-only library)
ExternalProject_Get_Property(rapidjson source_dir)
set(RAPIDJSON_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/vendor/rapidjson/src/rapidjson/include/)

add_mlir_dialect_library(MLIRAutoHogImporter
    AutoHogImporter.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/

    DEPENDS
    rapidjson

    DEPENDS
    rapidjson

    LINK_LIBS PUBLIC
    MLIRCGGI
    MLIRLWE
    LLVMSupport
    MLIRIR
    MLIRSupport
    MLIRArithDialect
    MLIRTensorDialect
    MLIRTranslateLib
)
target_include_directories(MLIRAutoHogImporter PUBLIC ${RAPIDJSON_INCLUDE_DIR})
