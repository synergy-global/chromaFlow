cmake_minimum_required(VERSION 3.20)

# Project definition
project(ChromaFlow 
    VERSION 1.0.0
    DESCRIPTION "Audio processing and machine learning library"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Determine if we are the top-level project
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(CHROMAFLOW_IS_TOP_LEVEL ON)
else()
    set(CHROMAFLOW_IS_TOP_LEVEL OFF)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ==============================================================================
# Package Managers Setup
# ==============================================================================



# Download CPM if not available
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")
    file(DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake
    )
endif()

include(cmake/CPM.cmake)
# 2. Conan Setup (if conandata.txt or conanfile exists)
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake")
    include(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup(TARGETS)
    message(STATUS "Conan integration enabled")
endif()

# Alternative Conan 2.0 setup
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/conan_toolchain.cmake")
    include(${CMAKE_CURRENT_BINARY_DIR}/conan_toolchain.cmake)
    message(STATUS "Conan 2.0 integration enabled")
endif()

# 3. FetchContent Setup
include(FetchContent)

# ==============================================================================
# Dependencies via CPM
# ==============================================================================

# Eigen3 via CPM (as requested)
CPMAddPackage(
    NAME Eigen3
    VERSION 3.4.0
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    OPTIONS
        "EIGEN_BUILD_DOC OFF"
        "EIGEN_BUILD_TESTING OFF"
        "EIGEN_BUILD_PKGCONFIG OFF"
)

# Optional: Add more dependencies via CPM
# Testing frameworks
option(CHROMAFLOW_BUILD_TESTS "Build ChromaFlow tests" OFF)
option(CHROMAFLOW_USE_CATCH2 "Use Catch2 testing framework" ON)

if(CHROMAFLOW_BUILD_TESTS)
    if(CHROMAFLOW_USE_CATCH2)
        # Catch2 testing framework
        CPMAddPackage(
            NAME Catch2
            GITHUB_REPOSITORY catchorg/Catch2
            VERSION 3.4.0
            OPTIONS
                "CATCH_INSTALL_DOCS OFF"
                "CATCH_INSTALL_EXTRAS ON"
        )
        # Provide Catch2's CMake module (for catch_discover_tests)
        if(Catch2_SOURCE_DIR)
            list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
        endif()
    else()
        # Google Test for testing (alternative)
        CPMAddPackage(
            NAME googletest
            GITHUB_REPOSITORY google/googletest
            VERSION 1.14.0
            OPTIONS
                "INSTALL_GTEST OFF"
                "gtest_force_shared_crt ON"
        )
    endif()
endif()

# Example: JSON library via CPM
option(CHROMAFLOW_USE_JSON "Use nlohmann/json library" OFF)
if(CHROMAFLOW_USE_JSON)
    CPMAddPackage(
        NAME nlohmann_json
        VERSION 3.11.2
        GITHUB_REPOSITORY nlohmann/json
        OPTIONS
            "JSON_BuildTests OFF"
    )
endif()


# ==============================================================================
# ChromaFlow Library Target
# ==============================================================================

# Collect all header files
file(GLOB_RECURSE CHROMAFLOW_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/chromaLib/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/chromaLib/*.hpp"
)

# Create header-only library
add_library(ChromaFlow INTERFACE)

add_custom_target(ChromaFlowHeaders SOURCES ${FUSION_DSP_HEADERS})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/chromaLIB FILES ${FUSION_DSP_HEADERS})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_target_properties(ChromaFlowHeaders PROPERTIES FOLDER "ChromaFlow")
# Add include directories
target_include_directories(ChromaFlow INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link Eigen3
target_link_libraries(ChromaFlow INTERFACE Eigen3::Eigen)

# Link Conan targets if available
if(TARGET CONAN_PKG::eigen)
    target_link_libraries(ChromaFlow INTERFACE CONAN_PKG::eigen)
endif()

# Optional dependencies
if(CHROMAFLOW_USE_JSON AND TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(ChromaFlow INTERFACE nlohmann_json::nlohmann_json)
endif()



    
if(APPLE)
    target_link_libraries(ChromaFlow INTERFACE "-framework Accelerate")
endif()
    

# Compiler features
target_compile_features(ChromaFlow INTERFACE cxx_std_17)


# ==============================================================================
# Testing (Optional)
# ==============================================================================
if(CHROMAFLOW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)

    # Convenience target: run Catch2 test binary directly (no CTest)
    if(CHROMAFLOW_IS_TOP_LEVEL)
        add_custom_target(tests
            COMMAND $<TARGET_FILE:chromaflow_tests>
            DEPENDS chromaflow_tests
        )
    endif()
endif()

# ==============================================================================
# Examples (Optional)
# ==============================================================================

option(CHROMAFLOW_BUILD_EXAMPLES "Build ChromaFlow examples" OFF)
if(CHROMAFLOW_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ==============================================================================
# Print Configuration Summary
# ==============================================================================

message(STATUS "")
message(STATUS "ChromaFlow Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build tests: ${CHROMAFLOW_BUILD_TESTS}")
if(CHROMAFLOW_BUILD_TESTS)
    message(STATUS "  Test framework: ${CHROMAFLOW_USE_CATCH2}")
endif()
message(STATUS "  Build examples: ${CHROMAFLOW_BUILD_EXAMPLES}")
message(STATUS "  Use JSON: ${CHROMAFLOW_USE_JSON}")
message(STATUS "  Use AudioFFT: ${CHROMAFLOW_USE_AUDIOFFT}")
message(STATUS "  Use FFTW: ${CHROMAFLOW_USE_FFTW}")
if(TARGET Eigen3::Eigen)
    message(STATUS "  Eigen3: Found via CPM")
elseif(TARGET CONAN_PKG::eigen)
    message(STATUS "  Eigen3: Found via Conan")
else()
    message(STATUS "  Eigen3: Not found")
endif()
message(STATUS "")
