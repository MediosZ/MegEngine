
file(GLOB_RECURSE SRCS impl/*.cpp include/*.h js/src/*.cpp js/src/*.h)

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
# add_compile_options(-pthread)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
#set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads REQUIRED)
#target_link_libraries(meg PRIVATE Threads::Threads)
# message(STATUS "use ${CMAKE_THREAD_LIBS_INIT}")

target_link_directories(meg PUBLIC "/usr/local/Cellar/gcc/10.2.0_4/lib/gcc/10/")
target_link_libraries(meg PUBLIC megengine_export)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/range-v3 ${PROJECT_BINARY_DIR}/third_party/range-v3)
target_link_libraries(meg PRIVATE range-v3)
target_compile_options(meg PRIVATE "-Wall")
set(FLAGS
    -s MODULARIZE=1
    -s FORCE_FILESYSTEM=1
    --bind
    -s EXPORT_NAME='MegJSModule'
    -s EXPORTED_RUNTIME_METHODS="['cwrap', 'ccall']"
    -s ALLOW_MEMORY_GROWTH=1
    -s INITIAL_MEMORY=512mb
    -s TOTAL_STACK=128mb
    -s DISABLE_EXCEPTION_CATCHING=0
    # -msse2
    # -msse
    # -msimd128
)
#-s USE_PTHREADS=1 
#-s PTHREAD_POOL_SIZE=4
#-s PROXY_TO_PTHREAD=1
#-s MALLOC=emmalloc
# --preload-file ${CMAKE_SOURCE_DIR}/data@/
string(REPLACE ";" " " FLAG_STR "${FLAGS}")
if(EMSCRIPTEN_ROOT_PATH)
set_target_properties(meg PROPERTIES
    LINK_FLAGS ${FLAG_STR}
)
endif()


add_dependencies(meg mgb_opdef)

# em++: error: '/usr/local/opt/emscripten/libexec/llvm/bin/wasm-ld -o meg.wasm CMakeFiles/meg.dir/impl/backward_graph_opt.cpp.o CMakeFiles/meg.dir/impl/blob_manager_impl.cpp.o CMakeFiles/meg.dir/impl/event_pool.cpp.o CMakeFiles/meg.dir/impl/interpreter/interpreter_impl.cpp.o CMakeFiles/meg.dir/impl/interpreter/profiler.cpp.o CMakeFiles/meg.dir/impl/op_def.cpp.o CMakeFiles/meg.dir/impl/op_trait.cpp.o CMakeFiles/meg.dir/impl/opr_utility.cpp.o CMakeFiles/meg.dir/impl/ops/atlas_runtime.cpp.o CMakeFiles/meg.dir/impl/ops/autogen.cpp.o CMakeFiles/meg.dir/impl/ops/backward_graph.cpp.o CMakeFiles/meg.dir/impl/ops/batch_norm.cpp.o CMakeFiles/meg.dir/impl/ops/broadcast.cpp.o CMakeFiles/meg.dir/impl/ops/cambricon_runtime.cpp.o CMakeFiles/meg.dir/impl/ops/collective_comm.cpp.o CMakeFiles/meg.dir/impl/ops/cond_take.cpp.o CMakeFiles/meg.dir/impl/ops/convolution.cpp.o CMakeFiles/meg.dir/impl/ops/deformable_conv2d.cpp.o CMakeFiles/meg.dir/impl/ops/deformable_psroi_pooling.cpp.o CMakeFiles/meg.dir/impl/ops/elemwise.cpp.o CMakeFiles/meg.dir/impl/ops/io_remote.cpp.o CMakeFiles/meg.dir/impl/ops/matrix_inverse.cpp.o CMakeFiles/meg.dir/impl/ops/nms.cpp.o CMakeFiles/meg.dir/impl/ops/opr_attr.cpp.o CMakeFiles/meg.dir/impl/ops/resize.cpp.o CMakeFiles/meg.dir/impl/ops/specializations.cpp.o CMakeFiles/meg.dir/impl/ops/tensor_manip.cpp.o CMakeFiles/meg.dir/impl/ops/tensorrt_runtime.cpp.o CMakeFiles/meg.dir/impl/ops/utility.cpp.o CMakeFiles/meg.dir/impl/ops/vision.cpp.o CMakeFiles/meg.dir/impl/ops/warp_affine.cpp.o CMakeFiles/meg.dir/impl/physical_tensor.cpp.o CMakeFiles/meg.dir/impl/profiler.cpp.o CMakeFiles/meg.dir/impl/proxy_graph.cpp.o CMakeFiles/meg.dir/impl/proxy_graph/proxy_graph.cpp.o CMakeFiles/meg.dir/impl/proxy_graph_detail.cpp.o CMakeFiles/meg.dir/impl/tensor_sanity_check.cpp.o CMakeFiles/meg.dir/js/src/engine.cpp.o CMakeFiles/meg.dir/js/src/grad.cpp.o CMakeFiles/meg.dir/js/src/grad_override.cpp.o CMakeFiles/meg.dir/js/src/tensor.cpp.o CMakeFiles/meg.dir/js/src/webassembly.cpp.o -L/usr/local/Cellar/gcc/10.2.0_4/lib/gcc/10 ../src/libmegengine_export.a ../flatbuffers/libflatbuffers.a /usr/local/Cellar/gcc/10.2.0_4/lib/gcc/10/libatomic.a -L/usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/crtbegin.o --whole-archive /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/libembind-rtti.a --no-whole-archive /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/libgl-mt.a /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/libal.a /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/libhtml5.a /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/libc-mt.a /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/libcompiler_rt-mt.a /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/libc++-mt.a /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/libc++abi-mt.a /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/libdlmalloc-mt.a /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/libc_rt_wasm.a /usr/local/Cellar/emscripten/2.0.22/libexec/cache/sysroot/lib/wasm32-emscripten/libsockets-mt.a -mllvm -combiner-global-alias-analysis=false -mllvm -enable-emscripten-cxx-exceptions -mllvm -enable-emscripten-sjlj -mllvm -disable-lsr --allow-undefined --import-memory --shared-memory --strip-debug --export main --export stackSave --export stackRestore --export stackAlloc --export __wasm_call_ctors --export __errno_location --export __emscripten_pthread_data_constructor --export __pthread_tsd_run_dtors --export _emscripten_call_on_thread --export _emscripten_do_dispatch_to_thread --export _emscripten_main_thread_futex --export _emscripten_thread_init --export emscripten_current_thread_process_queued_calls --export _emscripten_allow_main_runtime_queued_calls --export emscripten_futex_wake --export emscripten_get_global_libc --export emscripten_main_browser_thread_id --export emscripten_main_thread_process_queued_calls --export emscripten_register_main_browser_thread_id --export emscripten_run_in_main_runtime_thread_js --export emscripten_stack_set_limits --export emscripten_sync_run_in_main_thread_2 --export emscripten_sync_run_in_main_thread_4 --export emscripten_tls_init --export pthread_self --export memalign --export malloc --export free --export __cxa_is_pointer_type --export __cxa_can_catch --export setThrew --export _get_tzname --export _get_daylight --export _get_timezone --export-if-defined=__start_em_asm --export-if-defined=__stop_em_asm --export-table -z stack-size=134217728 --initial-memory=536870912 --no-entry --max-memory=536870912 --global-base=1024' failed (returned 1)