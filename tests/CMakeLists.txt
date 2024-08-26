# heir-check will run the lit based tests (add option -a for lit to show the details of tests and/or failures)
add_custom_target(check-heir
    COMMAND CMAKE_HEIR_PATH=${CMAKE_BINARY_DIR}/bin:${LLVM_TOOLS_BINARY_DIR} RUNFILES_DIR=/tmp lit  ${CMAKE_SOURCE_DIR}/tests/ -v
    COMMENT "Running lit test script (you need to be in virtualenv)"
    DEPENDS heir-opt
)
