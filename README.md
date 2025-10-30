# ChromaFlow

Audio processing and machine learning library with modern C++ and CMake.

## Package Management

This project supports multiple package managers for maximum flexibility:

### 1. CPM (CMake Package Manager) - Primary
- **Eigen3**: Linear algebra library (managed via CPM)
- Automatic download and configuration
- No external dependencies required

### 2. Conan - Optional
- Alternative package management
- Supports more complex dependency scenarios
- Requires Conan installation

### 3. FetchContent - For specific cases
- Git-based dependencies
- Custom build configurations

## Building

### Quick Start (CPM only)
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Using the Build Script
```bash
# Basic build
./build.sh

# Debug with tests (using Catch2)
./build.sh --debug --tests

# Using Google Test instead of Catch2
./build.sh --tests --gtest

# Build with AudioFFT support
./build.sh --audiofft

# Using Conan for dependencies
./build.sh --conan

# Clean build
./build.sh --clean
```

### Manual CMake Configuration

#### Option 1: CPM (Default)
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

#### Option 2: With Conan
```bash
# Install dependencies
conan install . --output-folder=build --build=missing

# Configure and build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CHROMAFLOW_BUILD_TESTS` | OFF | Build unit tests |
| `CHROMAFLOW_USE_CATCH2` | ON | Use Catch2 testing framework (vs Google Test) |
| `CHROMAFLOW_BUILD_EXAMPLES` | OFF | Build example applications |
| `CHROMAFLOW_USE_JSON` | OFF | Include nlohmann/json library |
| `CHROMAFLOW_USE_AUDIOFFT` | OFF | Include AudioFFT library |
| `CHROMAFLOW_USE_FFTW` | OFF | Include FFTW library |

## Dependencies

### Core Dependencies (via CPM)
- **Eigen3 3.4.0**: Linear algebra and matrix operations

### Optional Dependencies
- **Catch2 3.4.0**: Modern C++ unit testing framework (default)
- **Google Test**: Alternative unit testing framework
- **AudioFFT**: Simple C++ wrapper for FFT libraries
- **nlohmann/json**: JSON parsing and serialization
- **FFTW**: Fast Fourier Transform library

## Project Structure

```
chromaFlow/
├── chromaLib/              # Header-only library
│   ├── ChromaBaseClasses.h
│   ├── ChromaFeatureExtractor.h
│   ├── ChromaLayers.h
│   ├── ChromaLossFunctions.h
│   └── ChromaOptimizers.h
├── cmake/                  # CMake modules
│   ├── CPM.cmake
│   └── ChromaFlowConfig.cmake.in
├── CMakeLists.txt          # Main CMake configuration
├── conanfile.txt           # Conan dependencies
├── build.sh                # Build script
└── README.md
```

## Usage

### Including in Your Project

#### As a CMake Subdirectory
```cmake
add_subdirectory(chromaFlow)
target_link_libraries(your_target ChromaFlow)
```

#### As an Installed Package
```cmake
find_package(ChromaFlow REQUIRED)
target_link_libraries(your_target ChromaFlow::ChromaFlow)
```

### C++ Code Example
```cpp
#include "chromaLib/ChromaBaseClasses.h"

using namespace ChromaFlow;

int main() {
    // Create audio tensor
    AudioTensor audio;
    audio.numSamples = 1024;
    audio.numChannels = 2;
    audio.data = Eigen::VectorXf::Random(1024 * 2);
    
    // Use ChromaFlow classes...
    return 0;
}
```

## Requirements

- CMake 3.20 or higher
- C++17 compatible compiler
- Internet connection (for automatic dependency download)

### Optional
- Conan 1.x or 2.x (for Conan-based dependency management)

## License

[Add your license information here]