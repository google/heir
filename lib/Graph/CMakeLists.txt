get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

function(make_heir_exec  TARGET_NAME SOURCES)
        add_llvm_executable(${TARGET_NAME} PARTIAL_SOURCES_INTENDED ${SOURCES})
        llvm_update_compile_flags(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PUBLIC
        ${dialect_libs} ${conversion_libs}
        MLIROptLib MLIRSupport LLVMSupport /usr/local/lib/libgtest.a) # /usr/local/lib/libgtest_main.a)
endfunction()

#FIXME: linking against gtest is funny in Mac OSX
#make_heir_exec(graphtest GraphTest.cpp)
