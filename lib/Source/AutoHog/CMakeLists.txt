include(ExternalProject)

# Get RapidJson commit hash from bazel WORKSPACE file
file(READ "${CMAKE_SOURCE_DIR}/WORKSPACE" WORKSPACE_CONTENTS)
string(REGEX MATCH "name = \"rapidjson\"[^\n]*\n[^\n]*\n[^\n]*commit = \"([a-f0-9]+)\"" _ ${WORKSPACE_CONTENTS})
set(RAPIDJSON_COMMIT_HASH ${CMAKE_MATCH_1})
if(NOT RAPIDJSON_COMMIT_HASH)
    message(FATAL_ERROR "The rapidjson commit hash could not be read from WORKSPACE file.\
    Most likely the number of lines between the name and commit in the rapidjson block\
    there has changed and the number of \[^\\n\]*\\n in the REGEX here needs to be adjusted.")
else()
    message(STATUS "RapidJSON commit hash: ${RAPIDJSON_COMMIT_HASH}")
endif()

# Download RapidJSON
ExternalProject_Add(
    rapidjson
    PREFIX "vendor/rapidjson"
    GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
    GIT_TAG ${RAPIDJSON_COMMIT_HASH}
    TIMEOUT 10
    CMAKE_ARGS
        -DRAPIDJSON_BUILD_TESTS=OFF
        -DRAPIDJSON_BUILD_DOC=OFF
        -DRAPIDJSON_BUILD_EXAMPLES=OFF
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL
)

# Prepare RapidJSON (RapidJSON is a header-only library)
ExternalProject_Get_Property(rapidjson source_dir)
set(RAPIDJSON_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/vendor/rapidjson/src/rapidjson/include/)

add_mlir_library(HEIRAutoHogImporter
    AutoHogImporter.cpp

    DEPENDS
    rapidjson

    LINK_LIBS PUBLIC
    HEIRCGGI
    HEIRLWE
    LLVMSupport
    MLIRIR
    MLIRSupport
    MLIRArithDialect
    MLIRTensorDialect
    MLIRTranslateLib
)
target_include_directories(HEIRAutoHogImporter PUBLIC ${RAPIDJSON_INCLUDE_DIR})
target_link_libraries(HEIRSource INTERFACE  HEIRAutoHogImporter)
