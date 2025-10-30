#!/bin/bash

# ChromaFlow Build Script
# This script demonstrates how to build the project with different package managers

set -e

BUILD_DIR="build"
BUILD_TYPE="Release"
USE_CONAN=false
BUILD_TESTS=false
BUILD_EXAMPLES=false
USE_CATCH2=true
USE_AUDIOFFT=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --conan)
            USE_CONAN=true
            shift
            ;;
        --tests)
            BUILD_TESTS=true
            shift
            ;;
        --examples)
            BUILD_EXAMPLES=true
            shift
            ;;
        --gtest)
            USE_CATCH2=false
            shift
            ;;
        --catch2)
            USE_CATCH2=true
            shift
            ;;
        --audiofft)
            USE_AUDIOFFT=true
            shift
            ;;
        --clean)
            echo "Cleaning build directory..."
            rm -rf $BUILD_DIR
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --debug      Build in Debug mode (default: Release)"
            echo "  --conan      Use Conan for dependency management"
            echo "  --tests      Build tests"
            echo "  --examples   Build examples"
            echo "  --catch2     Use Catch2 testing framework (default)"
            echo "  --gtest      Use Google Test framework"
            echo "  --audiofft   Include AudioFFT library"
            echo "  --clean      Clean build directory before building"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Building ChromaFlow..."
echo "Build type: $BUILD_TYPE"
echo "Use Conan: $USE_CONAN"
echo "Build tests: $BUILD_TESTS"
if [ "$BUILD_TESTS" = true ]; then
    if [ "$USE_CATCH2" = true ]; then
        echo "Test framework: Catch2"
    else
        echo "Test framework: Google Test"
    fi
fi
echo "Build examples: $BUILD_EXAMPLES"
echo "Use AudioFFT: $USE_AUDIOFFT"

# Create build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with Conan if requested
if [ "$USE_CONAN" = true ]; then
    echo "Installing dependencies with Conan..."
    conan install .. --output-folder=. --build=missing -s build_type=$BUILD_TYPE
fi

# Configure CMake
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE
    -DCHROMAFLOW_BUILD_TESTS=$BUILD_TESTS
    -DCHROMAFLOW_BUILD_EXAMPLES=$BUILD_EXAMPLES
    -DCHROMAFLOW_USE_CATCH2=$USE_CATCH2
    -DCHROMAFLOW_USE_AUDIOFFT=$USE_AUDIOFFT
)

if [ "$USE_CONAN" = true ]; then
    CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake)
fi

echo "Configuring CMake..."
cmake .. "${CMAKE_ARGS[@]}"

# Build
echo "Building..."
cmake --build . --config $BUILD_TYPE -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build completed successfully!"

# Run tests if built
if [ "$BUILD_TESTS" = true ]; then
    echo "Running tests..."
    ctest --output-on-failure
fi