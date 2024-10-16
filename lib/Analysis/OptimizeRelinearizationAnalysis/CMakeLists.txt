
# Get Or-Tools and its dependencies
message(CHECK_START "Fetching Or-Tools")
list(APPEND CMAKE_MESSAGE_INDENT " -- ")
  # Get Or-Tools commit hash from bazel WORKSPACE file
  file(READ "${CMAKE_SOURCE_DIR}/WORKSPACE" WORKSPACE_CONTENTS)
  string(REGEX MATCH "name = \"com_google_ortools\"[^\n]*\n[^\n]*commit = \"([a-f0-9]+)\"" _ ${WORKSPACE_CONTENTS})
  set(ORTOOLS_COMMIT_HASH ${CMAKE_MATCH_1})
  if(NOT ORTOOLS_COMMIT_HASH)
      message(FATAL_ERROR "The or-tools commit hash could not be read from WORKSPACE file.\
        Most likely the number of lines between the name and commit in the or-tools block\
        there has changed and the number of \[^\\n\]*\\n in the REGEX here needs to be adjusted.")
  else()
    message(STATUS "or-tools commit hash: ${ORTOOLS_COMMIT_HASH}")
  endif()


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
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "fetched")


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
