# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("chatapp")

# Download nlohmann/json single-header at configure time if not already present
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/json.hpp)
    file(DOWNLOAD
        https://raw.githubusercontent.com/nlohmann/json/55f93686c01528224f448c19128836e7df245f72/single_include/nlohmann/json.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/json.hpp
    )
endif()

set(GENIE_HEADERS_PATH "${QNN_SDK_ROOT_PATH}/include/Genie")
set(GENIE_LIB_PATH "${QNN_SDK_ROOT_PATH}/lib/aarch64-android/libGenie.so")

include_directories("includes")

# Include Genie headers here
include_directories(${GENIE_HEADERS_PATH})

add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        PromptHandler.cpp
        GenieWrapper.cpp
        GenieLib.cpp)

# Link libGenie.so here
target_link_libraries(${CMAKE_PROJECT_NAME}
        "${GENIE_LIB_PATH}"
        android
        log)
