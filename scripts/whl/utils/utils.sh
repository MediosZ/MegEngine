#!/usr/bin/env bash
set -e

OS=$(uname -s)

docker_file=""
function config_docker_file() {
    case $(uname -m) in
        x86_64) docker_file=Dockerfile ;;
        aarch64) docker_file=Dockerfile_aarch64 ;;
        *) echo "nonsupport env!!!";exit -1 ;;
    esac
}

function ninja_dry_run_and_check_increment() {
    if [ $# -eq 3 ]; then
        _BUILD_SHELL=$1
        _BUILD_FLAGS="$2 -n"
        _INCREMENT_KEY_WORDS=$3
    else
        echo "err call ninja_dry_run_and_check_increment"
        exit -1
    fi

    ${_BUILD_SHELL} ${_BUILD_FLAGS} 2>&1 | tee dry_run.log

    DIRTY_LOG=`cat dry_run.log`
    if [[ "${DIRTY_LOG}" =~ ${_INCREMENT_KEY_WORDS} ]]; then
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "python3 switch increment build failed, some MR make a wrong CMakeLists.txt depends"
        echo "or build env can not find default python3 in PATH env"
        echo "please refs for PYTHON3_EXECUTABLE_WITHOUT_VERSION define at SRC_ROOT/CMakeLists.txt"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        exit -1
    fi
}

PYTHON_API_INCLUDES=""

function check_build_ninja_python_api() {
    INCLUDE_KEYWORD=""
    IS_MINOR_HIT=FALSE
    if [ $# -eq 1 ]; then
        ver=$1
        echo "org args: ${ver}"
        if [[ $OS =~ "NT" ]]; then
            INCLUDE_KEYWORD="${ver}\\\\include"
            PYTHON_API_INCLUDES="3.5.4\\\\include 3.6.8\\\\include 3.7.7\\\\include 3.8.3\\\\include"
        elif [[ $OS =~ "Linux" ]]; then
            INCLUDE_KEYWORD="include/python3.${ver:1:1}"
            PYTHON_API_INCLUDES="include/python3.5 include/python3.6 include/python3.7 include/python3.8"
        elif [[ $OS =~ "Darwin" ]]; then
            INCLUDE_KEYWORD="include/python3.${ver:2:1}"
            PYTHON_API_INCLUDES="include/python3.5 include/python3.6 include/python3.7 include/python3.8"
        else
            echo "unknown OS: ${OS}"
            exit -1
        fi
    else
        echo "err call check_build_ninja_python_api"
        exit -1
    fi
    echo "try check python INCLUDE_KEYWORD: ${INCLUDE_KEYWORD} is invalid in ninja.build or not"

    NINJA_BUILD=`cat build.ninja`
    for PYTHON_API_INCLUDE in ${PYTHON_API_INCLUDES}
    do
        echo "check PYTHON_API_INCLUDE vs INCLUDE_KEYWORD : (${PYTHON_API_INCLUDE} : ${INCLUDE_KEYWORD})"
        if [ ${PYTHON_API_INCLUDE} = ${INCLUDE_KEYWORD} ]; then
            if [[ "${NINJA_BUILD}" =~ ${PYTHON_API_INCLUDE} ]]; then
                echo "hit INCLUDE_KEYWORD: ${INCLUDE_KEYWORD} in build.ninja"
                IS_MINOR_HIT="TRUE"
            else
                echo "Err happened can not find INCLUDE_KEYWORD: ${INCLUDE_KEYWORD} in build.ninja"
                exit -1
            fi
        else
            if [[ "${NINJA_BUILD}" =~ ${PYTHON_API_INCLUDE} ]]; then
                echo "Err happened: find PYTHON_API_INCLUDE: ${PYTHON_API_INCLUDE} in build.ninja"
                echo "But now INCLUDE_KEYWORD: ${INCLUDE_KEYWORD}"
                exit -1
            fi
        fi
    done

    if [ ${IS_MINOR_HIT} = "FALSE" ]; then
        echo "Err happened, can not hit any MINOR api in ninja.build"
        exit -1
    fi
}

function check_python_version_is_valid() {
    want_build_version=$1
    support_version=$2
    if [ $# -eq 2 ]; then
        ver=$1
    else
        echo "err call check_python_version_is_valid"
        exit -1
    fi
    for i_want_build_version in ${want_build_version}
    do
        is_valid="false"
        for i_support_version in ${support_version}
        do
            if [ ${i_want_build_version} == ${i_support_version} ];then
                is_valid="true"
            fi
        done
        if [ ${is_valid} == "false" ];then
            echo "invalid build python version : \"${want_build_version}\", now support party of \"${support_version}\""
            exit -1
        fi
    done
}
