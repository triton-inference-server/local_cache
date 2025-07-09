# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.31.8)
project(tritonlocalcache LANGUAGES C CXX)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
option(TRITON_ENABLE_METRICS "Include metrics support in cache" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent's composability isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core)

#
# Shared library implementing the Triton Cache API
#
configure_file(src/libtritoncache_local.ldscript libtritoncache_local.ldscript COPYONLY)

add_library(
  triton-local-cache SHARED
  src/cache_api.cc
  src/local_cache.cc
)

add_library(
  TritonLocalCache::triton-local-cache ALIAS triton-local-cache
)

target_include_directories(
  triton-local-cache
  PUBLIC
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(triton-local-cache PRIVATE cxx_std_17)
target_compile_options(
  triton-local-cache PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

if(${TRITON_ENABLE_METRICS})
  target_compile_definitions(
    triton-local-cache
    PRIVATE TRITON_ENABLE_METRICS=1
  )
endif() # TRITON_ENABLE_METRICS

if(${TRITON_ENABLE_LOGGING})
  target_compile_definitions(
    triton-local-cache
    PRIVATE TRITON_ENABLE_LOGGING=1
  )
endif() # TRITON_ENABLE_LOGGING

target_link_libraries(
  triton-local-cache
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-cacheapi    # from repo-core
    triton-core-serverstub  # from repo-core
    triton-common-logging   # from repo-common
)

# TODO: Windows support
set_target_properties(
  triton-local-cache PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME tritoncache_local
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtritoncache_local.ldscript
  LINK_FLAGS "-Wl,--version-script libtritoncache_local.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonLocalCache)

install(
  TARGETS
    triton-local-cache
  EXPORT
    triton-local-cache-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/caches/local
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/caches/local
)

install(
  EXPORT
    triton-local-cache-targets
  FILE
    TritonLocalCacheTargets.cmake
  NAMESPACE
    TritonLocalCache::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonLocalCacheConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonLocalCacheConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonLocalCacheConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-local-cache-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonLocalCacheTargets.cmake
  NAMESPACE TritonLocalCache::
)

export(PACKAGE TritonLocalCache)
