# ChromaFlow

Tiny, RT‑aware neural DSP and feature extraction library in modern C++17 with CMake. It provides minimal differentiable building blocks (dense, conv, RNN, attention), simple IIR‑style online learning, and a real‑time safe base class for audio processing.

## Highlights
- Minimal differentiable modules: Dense, 1D Conv, RNNCell, Attention, LayerNorm
- Tiny autograd: local IIR error memory + gradient smoothing, hard weight clipping
- Real‑time safety: allocation‑free audio path with preallocated buffers and optional async adaptation
- Feature extraction: MFCC and core spectral features using Eigen and AudioFFT (optional)
- Header‑only chromaLib for easy integration

## Architecture
- Core tensors and base classes: chromaLib/ChromaBaseClasses.h
- Layers: chromaLib/ChromaLayers.h
- Losses: chromaLib/ChromaLossFunctions.h
- Optimizers: chromaLib/ChromaOptimizers.h
- Feature extraction: chromaLib/ChromaFeatureExtractor.h

### Real‑Time Path
- NeuralDSPLayer manages the audio thread contract:
  - configureRT(sampleRate, batchSize, controlStride, asyncAdapt)
  - processBlock(...) performs bounded work only (no allocations)
  - Optional SPSC queue for off‑thread adapt; call drainAdaptQueue() on a control thread

## Build

### Quick Start (CPM)
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### Using the Build Script
```bash
# Basic build
./build.sh

# Debug with tests (Catch2)
./build.sh --debug --tests

# Google Test instead of Catch2
./build.sh --tests --gtest

# Enable optional components
./build.sh --audiofft

# Conan toolchain
./build.sh --conan

# Clean
./build.sh --clean
```

### Manual CMake + Conan
```bash
conan install . --output-folder=build --build=missing
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| CHROMAFLOW_BUILD_TESTS | OFF | Build unit tests |
| CHROMAFLOW_USE_CATCH2 | ON | Use Catch2 vs Google Test |
| CHROMAFLOW_BUILD_EXAMPLES | OFF | Build examples |
| CHROMAFLOW_USE_AUDIOFFT | OFF | AudioFFT support |
| CHROMAFLOW_USE_FFTW | OFF | FFTW support |

## Project Structure
```
chromaFlow/
├── chromaLib/
│   ├── ChromaBaseClasses.h
│   ├── ChromaFeatureExtractor.h
│   ├── ChromaLayers.h
│   ├── ChromaLossFunctions.h
│   └── ChromaOptimizers.h
├── tests/
├── cmake/
├── CMakeLists.txt
├── build.sh
└── README.md
```

## Quick Examples

### Online learning with DenseLayer
```cpp
#include "chromaLib/ChromaLayers.h"
#include "chromaLib/ChromaLossFunctions.h"
using namespace ChromaFlow;
using namespace ChromaFlow::Layers;
using namespace ChromaFlow::LossFunctions;

DenseLayer dense(/*in*/8, /*out*/4);

FeatureTensor x; x.numSamples=1; x.features=8;
x.data = Eigen::MatrixXf::Constant(1,8, 0.1f);

FeatureTensor y = dense.forward(x);

MSELoss loss;
FeatureTensor target; target.numSamples=1; target.features=4;
target.data = Eigen::MatrixXf::Zero(1,4);

FeatureTensor grad = loss.calculate(y, target);
dense.learn(grad);
```

### Real‑time processing skeleton
```cpp
#include "chromaLib/ChromaBaseClasses.h"
#include "chromaLib/ChromaLayers.h"
using namespace ChromaFlow;
using namespace ChromaFlow::Layers;

class MyDSP : public NeuralDSPLayer {
public:
  void prepare(double sr) override {
    configureRT(sr, /*batchSize*/1, /*controlStride*/64, /*asyncAdapt*/true);
    layer = std::make_unique<DenseLayer>(1, 1);
  }
  FeatureTensor adapt(const FeatureTensor& input) override {
    auto y = layer->forward(input);
    // compute grad and layer->learn(grad)...
    return y;
  }
  float process(float in) override { return in; }
private:
  std::unique_ptr<DenseLayer> layer;
};

// audio thread: dsp.processBlock(in, out, n)
// control thread: dsp.drainAdaptQueue()
```

### Feature extraction
```cpp
#include "chromaLib/ChromaFeatureExtractor.h"
using namespace ChromaFlow;

FeatureExtractor fx;
FeatureExtractor::Layout L = fx.getLayout();
AudioTensor a; a.numSamples=1024; a.numChannels=1; a.data = Eigen::VectorXf::Random(1024);
FeatureTensor feats = fx.extractFeatures(a);
```

## Testing
```bash
./build.sh --tests --debug
./build/tests/chromaflow_tests
```
When using ctest directly, you may see messages from external packages; running the test binary is recommended for clarity.

## RT Safety Guidelines
- No allocations or locks on the audio thread; use configureRT and preallocated buffers
- Move adaptation off the audio thread with async queue and drainAdaptQueue on a control thread
- Keep per‑sample work bounded; avoid I/O and logging in process/processBlock

## Requirements
- CMake 3.20+
- C++17 compiler
- Internet connection for CPM fetches (if not vendored)

### Optional
- Conan 1.x/2.x for dependency management

## License
Add your license here.
