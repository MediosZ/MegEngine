# mgb tablegen executable
set(TABLE_TARGET mgb-mlir-autogen)
set(MGE_IR_DIR_STR ${MGE_IR_DIR}/)
configure_file(./setupfs.h.in ./setupfs.h)
file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${TABLE_TARGET} ${SRCS})
target_include_directories(${TABLE_TARGET} PRIVATE ${MLIR_LLVM_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${TABLE_TARGET} PRIVATE LLVMTableGen MLIRTableGen LLVMSupport)
set(FLAGS
    -lnodefs.js
)
string(REPLACE ";" " " FLAG_STR "${FLAGS}")
set_target_properties(${TABLE_TARGET} PROPERTIES
CXX_STANDARD 17
LINK_FLAGS ${FLAG_STR}
)
set(MGB_TABLEGEN_EXE ${TABLE_TARGET})


# generate megbrain opdef c header and python bindings
set(LLVM_TARGET_DEFINITIONS ${MGE_IR_DIR}/ops.td)
tablegen(MGB opdef.h.inl ${MGE_IR_INCLUDE_DIRS} "--gen-cpp-header")
tablegen(MGB opdef.cpp.inl ${MGE_IR_INCLUDE_DIRS} "--gen-cpp-body")
# tablegen(MGB opdef.py.inl ${MGE_IR_INCLUDE_DIRS} "--gen-python-binding")
# tablegen(MGB opdef.cpy.inl ${MGE_IR_INCLUDE_DIRS} "--gen-python-c-extension")
add_custom_target(mgb_opdef ALL DEPENDS opdef.h.inl opdef.cpp.inl param_defs_tblgen)
set(MGB_OPDEF_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
