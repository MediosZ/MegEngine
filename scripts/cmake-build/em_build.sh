#!/usr/bin/env bash
set -e
BUILD_TYPE=Release
MGE_BUILD_PC=OFF
MGE_ARCH=naive
REMOVE_OLD_BUILD=false

while getopts "rdc" arg
do
    case $arg in
        d)
            echo "Build with Debug mode"
            BUILD_TYPE=Debug
            ;;
        c)
            echo "Build for PC"
            MGE_BUILD_PC=ON
            ;;
        r)
            echo "config REMOVE_OLD_BUILD=true"
            REMOVE_OLD_BUILD=true
            ;;
        ?)
            echo "unkonw argument"
            ;;
    esac
done

echo "EXTRA_CMAKE_ARGS: ${EXTRA_CMAKE_ARGS}"
echo "------------------------------------"
echo "build config summary:"
echo "BUILD_TYPE: $BUILD_TYPE"
echo "MGE_BUILD_PC: $MGE_BUILD_PC"
echo "MGE_INFERENCE_ONLY: $MGE_INFERENCE_ONLY"
echo "------------------------------------"

READLINK=greadlink
SRC_DIR=$($READLINK -f "`dirname $0`/../../")
source $SRC_DIR/scripts/cmake-build/utils/utils.sh

ROOT_DIR=$SRC_DIR/build_dir/emscripten

CMAKE="emcmake cmake"
MAKE="emmake make"
if [ $MGE_BUILD_PC = "ON" ];then
    ROOT_DIR=$SRC_DIR/build_dir/emscriptenc
    CMAKE="cmake"
    MAKE="make"
fi

if [ $BUILD_TYPE = "Debug" ];then
    ROOT_DIR=${ROOT_DIR}d
fi

BUILD_DIR=$ROOT_DIR/build
INSTALL_DIR=$ROOT_DIR/install


echo "build dir: $BUILD_DIR"
echo "install dir: $INSTALL_DIR"
echo "build type: $BUILD_TYPE"
try_remove_old_build $REMOVE_OLD_BUILD $BUILD_DIR $INSTALL_DIR
echo "create build dir"
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
$CMAKE \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DMGE_INFERENCE_ONLY=OFF\
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DMGE_WITH_CUDA=OFF \
    -DMGE_BUILD_IMPERATIVE_RT=OFF \
    -DMGE_BUILD_SDK=OFF \
    -DMGE_BUILD_MEGJS=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DMGE_ENABLE_LOGGING=ON \
    -DLLVM_COMPILER_IS_GCC_COMPATIBLE=1 \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
    -DXNNPACK_LIBRARY_TYPE=static \
    -DMGE_WITH_TEST=OFF \
    -Wall \
    ${EXTRA_CMAKE_ARGS} \
    $SRC_DIR

$MAKE -j$(nproc)
# emmake make install/strip
