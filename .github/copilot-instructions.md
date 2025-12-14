# Copilot Instructions for ChromaFlow

These guidelines help AI coding agents work productively in this C++/CMake audio ML library.

## Architecture
- **Header-only library:** Core code lives in `chromaLib/` and is consumed via `target_link_libraries(ChromaFlow)`; no compiled sources.
- **Namespaces:** All public APIs are under `ChromaFlow`.
- **Primitives:**
  - `AudioTensor` (1D audio, `Eigen::VectorXf`), `FeatureTensor` (2D features, `Eigen::MatrixXf`), `ParamTensor` (named float params), `WeightTensor`.
  - `DifferentiableModule` base: `forward(...)`, optional overload with `ParamTensor`, `predictParams(...)`, `learn(...)`, `getNumParams()`, `reset()`.
- **Layers:** Implemented in `chromaLib/ChromaLayers.h`.
  - `convolutionalLayer`: 1D conv-like, trainables: `kernel_`, `biases_`.
  - `denseLayer`: EMA preconditioning + optional layer norm; trainables: `weights`, `gamma`, `beta`, `learnable_alpha`.
  - `attentionLayer`: single-vector and sequence paths; internal `denseLayer`s for Q/K/V; trainables: `Wq/Wk/Wv/Wo` plus sublayers.
  - `RNNCell`: tanh recurrent cell with `W_x`, `W_h`, `b`, maintains `hidden_state`.
- **Optimizers:** See `chromaLib/ChromaOptimizers.h` (e.g., `SGDWithMomentum`). Layers call `optimizer.update(...)` in `backward/learn`.
- **Analysis utilities:** `NNAnalyzer` in `ChromaBaseClasses.h` counts params and estimates memory (`calculateTotalMemory*`).

## Build & Test
- **Primary path (CPM):**
  - Configure/build: `mkdir build && cd build && cmake .. && cmake --build .`
  - Script: `./build.sh [--debug] [--tests] [--catch2|--gtest] [--audiofft] [--conan] [--clean]`
  - CMake options: `CHROMAFLOW_BUILD_TESTS` (OFF), `CHROMAFLOW_USE_CATCH2` (ON), `CHROMAFLOW_BUILD_EXAMPLES` (OFF), `CHROMAFLOW_USE_AUDIOFFT` (OFF by default).
- **Conan path (optional):** `conan install . --output-folder=build --build=missing` then CMake with `-DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake`.
- **Tests:** Catch2 by default. Test target defined in `tests/CMakeLists.txt` as `chromaflow_tests`. When `CHROMAFLOW_BUILD_TESTS=ON`, either run `ctest --output-on-failure` (via script) or run the test binary.
- **Compiler standard:** Project sets `CMAKE_CXX_STANDARD 23` but exports `cxx_std_17` on the interface; write code compatible with C++17 for consumers.

## Conventions & Patterns
- **Header-only style:** Add functionality in headers under `chromaLib/`; avoid adding `.cpp` unless the build is updated accordingly.
- **Eigen usage:** Prefer `Eigen::VectorXf` for 1D feature vectors and `Eigen::MatrixXf` for 2D; respect row/column shapes used in layers (often `1×N` rows for feature outputs).
- **Forward/Backward contract:**
  - `forward` typically emits a `FeatureTensor` with `numSamples=1` and `features=dim` for non-sequence layers.
  - `backward` returns `{grad_prev, act_prev}` and performs optimizer updates; `learn(...)` generally delegates to `backward(...)`.
  - Use `getNumParams()` to reflect trainable sizes; keep it consistent with tests (see `tests/test_chroma_base_classes.cpp`).
- **Attention single-vector path:** If `input.data.rows() <= 1`, use simplified attention path with softmax over elementwise `q*k`; multi-row path builds Q/K/V per head.
- **EMA in dense:** `learnable_alpha` clamped to `(0.001, 0.999)`; `input_ema_state` blends external input with alpha.
- **Clip and activation helpers:** Reuse small utilities in `ChromaBaseClasses.h` and local `ChromaUtils::clip`.

## External Dependencies
- **Eigen3:** Required; managed via CPM (or Conan). Linked as `Eigen3::Eigen`.
- **Catch2 / GoogleTest:** Optional, gated by `CHROMAFLOW_USE_CATCH2`.
- **AudioFFT:** Optional via `FetchContent` when `CHROMAFLOW_USE_AUDIOFFT=ON`.

## Working Examples
- **Parameter counting:** See `tests/test_chroma_base_classes.cpp` for expectations,
  e.g., dense layer params = `rows*cols + gamma.size() + beta.size() + alpha.size()`.
- **Model aggregates:** `NNAnalyzer::countTrainableParameters(modules)` sums `getNumParams()` across `std::shared_ptr<DifferentiableModule>`.
- **Memory estimates:** `NNAnalyzer::calculateTotalMemory(params, isAdam, bytes)` where Adam uses two state vectors.

## Practical Tips
- When adding new layers, implement `forward`, `backward`, `learn`, `getNumParams`, and `reset` consistently with existing layers.
- Keep gradients 1D or `N×1` as expected by current tests; check shapes before operations.
- If you introduce sources or install rules, update `CMakeLists.txt` minimally and mirror existing options/summary messages.
