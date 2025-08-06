#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# add_compile_definitions(DISABLE_BACKWARD)
# add_compile_definitions(DISABLE_SPLIT)
# add_compile_definitions(DISABLE_PAGEDKV)
# add_compile_definitions(DISABLE_APPENDKV)
# add_compile_definitions(DISABLE_LOCAL)
# add_compile_definitions(DISABLE_SOFTCAP)
# add_compile_definitions(DISABLE_PACKGQA) add_compile_definitions(DISABLE_FP8)
# add_compile_definitions(DISABLE_VARLEN)
# add_compile_definitions(DISABLE_HDIM64)
# add_compile_definitions(DISABLE_HDIM96)
# add_compile_definitions(DISABLE_HDIM192)
# add_compile_definitions(DISABLE_HDIM256) add_compile_definitions(DISABLE_SM8x)
# file(GLOB SRCS *.cpp flash-attention/hopper/*.cpp
# flash-attention/hopper/instantiations/*.cu)

set(CMAKE_VERBOSE_MAKEFILE ON)

file(GLOB SRCS *.cpp)
set(PLUGIN_SOURCES ${PLUGIN_SOURCES} ${SRCS})
set(PLUGIN_SOURCES
    ${PLUGIN_SOURCES}
    PARENT_SCOPE)
