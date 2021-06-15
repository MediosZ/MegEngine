file(GLOB_RECURSE SRCS impl/*.cpp include/*.h)

file(GLOB_RECURSE ALL_HEADERS
    ${PROJECT_SOURCE_DIR}/src/core/include/*
    ${PROJECT_SOURCE_DIR}/src/opr/include/*
    ${PROJECT_SOURCE_DIR}/src/serialization/include/*
    ${PROJECT_SOURCE_DIR}/src/plugin/include/*
    ${PROJECT_SOURCE_DIR}/dnn/include/*)

    add_subdirectory(tablegen)

add_executable(meg ${SRCS})
target_include_directories(meg PUBLIC include 
    ${PROJECT_SOURCE_DIR}/src/serialization/include
    ${PROJECT_SOURCE_DIR}/src/core/include
    ${MGB_OPDEF_OUT_DIR}
    )
target_link_directories(meg PUBLIC "/usr/local/Cellar/gcc/10.2.0_4/lib/gcc/10/")
target_link_libraries(meg PUBLIC megengine_export)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/range-v3 ${PROJECT_BINARY_DIR}/third_party/range-v3)
target_link_libraries(meg PRIVATE range-v3)

set(FLAGS
    -s MODULARIZE=1
    --bind
    -s EXPORT_NAME='MegJSModule'
    -s EXTRA_EXPORTED_RUNTIME_METHODS="['cwrap', 'ccall']"
    --preload-file ${CMAKE_SOURCE_DIR}/data@/
)
string(REPLACE ";" " " FLAG_STR "${FLAGS}")

set_target_properties(meg PROPERTIES
LINK_FLAGS ${FLAG_STR}
)
add_dependencies(meg mgb_opdef)
