# Get Or-Tools commit hash from bazel WORKSPACE file
file(READ "${CMAKE_SOURCE_DIR}/WORKSPACE" WORKSPACE_CONTENTS)
string(REGEX MATCH "name = \"com_google_ortools\"[^\n]*\n[^\n]*\n[^\n]*commit = \"([a-f0-9]+)\"" _ ${WORKSPACE_CONTENTS})
set(ORTOOLS_COMMIT_HASH ${CMAKE_MATCH_1})
message(DEBUG "or-tools commit hash: ${ORTOOLS_COMMIT_HASH}")

# Fetch Or-Tools
include(FetchContent)
FetchContent_Declare(
  or-tools
  GIT_REPOSITORY https://github.com/google/or-tools.git
  GIT_TAG        ${ORTOOLS_COMMIT_HASH}
  EXCLUDE_FROM_ALL
)
# TODO: Can we narrow down the or-tools dependencies/options we need?
set(BUILD_DEPS ON)
FetchContent_MakeAvailable(or-tools)

add_mlir_library(HEIROptimizeRelinearizationAnalysis
        OptimizeRelinearizationAnalysis.cpp

        LINK_LIBS PUBLIC
        HEIRBGV
        LLVMSupport
        MLIRAnalysis
        MLIRIR
        MLIRSupport
        ortools::ortools
)
target_link_libraries(HEIRAnalysis INTERFACE HEIROptimizeRelinearizationAnalysis)
