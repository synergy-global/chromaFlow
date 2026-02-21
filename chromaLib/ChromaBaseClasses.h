/**
 * @file ChromaBaseClasses.h
 * @brief Core tensor types, differentiable modules, optimizers and utilities.
 */
#pragma once

#include <vector>

#include <memory>
#include <cmath>
#include "Eigen/Dense"
#include <algorithm>
#include <map>
#include <cassert>
#include <atomic>
#include <cassert>
namespace ChromaFlow
{
    /**
     * @brief 2D audio tensor with shape (numSamples, numChannels).
     */
    struct AudioTensor
    {
        Eigen::ArrayXXf data;
        int numSamples;
        int numChannels;
    };

    /**
     * @brief 2D feature tensor with shape (numSamples, features).
     */
    struct FeatureTensor
    {
        Eigen::MatrixXf data;
        int numSamples;
        int features;
    };

    /// @name FeatureTensor arithmetic helpers

    /**
     * @brief Create a new FeatureTensor with the same shape as a reference.
     *
     * @param ref Reference tensor.
     * @return New tensor with identical shape.
     */
    inline FeatureTensor makeLike(const FeatureTensor &ref)
    {
        FeatureTensor out;
        out.numSamples = static_cast<int>(ref.data.rows());
        out.features = static_cast<int>(ref.data.cols());
        out.data.resize(ref.data.rows(), ref.data.cols());
        return out;
    }

    /**
     * @brief Check if all elements of a FeatureTensor are finite.
     *
     * @param t Tensor to check.
     * @return True if all elements are finite, false otherwise.
     */
    inline bool isFinite(const FeatureTensor &t)
    {
        return t.data.allFinite();
    }
    /**
     * @brief Add a FeatureTensor in-place.
     *
     * @param a Tensor to add to.
     * @param b Tensor.
     */
    inline void addInPlace(FeatureTensor &a, const FeatureTensor &b)
    {
        assert(a.data.rows() == b.data.rows());
        assert(a.data.cols() == b.data.cols());
        a.data += b.data;
    }
    /**
     * @brief Subtract a FeatureTensor in-place.
     *
     * @param a Tensor to subtract from.
     * @param b Tensor.
     */
    inline void subInPlace(FeatureTensor &a, const FeatureTensor &b)
    {
        assert(a.data.rows() == b.data.rows());
        assert(a.data.cols() == b.data.cols());
        a.data -= b.data;
    }
    /**
     * @brief Element-wise division of two FeatureTensors.
     *
     * Both tensors must have identical shapes.
     */
    inline FeatureTensor divInPlace(FeatureTensor &a, const FeatureTensor &b)
    {
        assert(a.data.rows() == b.data.rows());
        assert(a.data.cols() == b.data.cols());
        a.data.array() /= (b.data.array() + 1e-10f);
        return a;
    }
    /**
     * @brief Element-wise multiplication of two FeatureTensors.
     *
     * Both tensors must have identical shapes.
     */
    inline void mulInPlace(FeatureTensor &a, const FeatureTensor &b)
    {
        assert(a.data.rows() == b.data.rows());
        assert(a.data.cols() == b.data.cols());
        a.data.array() *= b.data.array();
    }
    /**
     * @brief Element-wise addition of two FeatureTensors.
     *
     * Both tensors must have identical shapes.
     */
    inline FeatureTensor operator+(const FeatureTensor &a,
                                   const FeatureTensor &b)
    {
        assert(a.data.rows() == b.data.rows());
        assert(a.data.cols() == b.data.cols());

        FeatureTensor out = makeLike(a);
        out.data = a.data + b.data;
        return out;
    }

    /**
     * @brief Element-wise subtraction of two FeatureTensors.
     *
     * Both tensors must have identical shapes.
     */
    inline FeatureTensor operator-(const FeatureTensor &a,
                                   const FeatureTensor &b)
    {
        assert(a.data.rows() == b.data.rows());
        assert(a.data.cols() == b.data.cols());

        FeatureTensor out = makeLike(a);
        out.data = a.data - b.data;
        return out;
    }

    /**
     * @brief Element-wise division of two FeatureTensors.
     *
     * Both tensors must have identical shapes.
     */
    inline FeatureTensor operator/(const FeatureTensor &a,
                                   const FeatureTensor &b)
    {
        assert(a.data.rows() == b.data.rows());
        assert(a.data.cols() == b.data.cols());

        FeatureTensor out = makeLike(a);
        out.data = a.data.array() /
                   (b.data.array() + 1e-10f);
        return out;
    }
    /**
     * @brief Element-wise multiplication of two FeatureTensors.
     *
     * Both tensors must have identical shapes.
     */
    inline FeatureTensor operator*(const FeatureTensor &a,
                                   const FeatureTensor &b)
    {
        assert(a.data.rows() == b.data.rows());
        assert(a.data.cols() == b.data.cols());

        FeatureTensor out = makeLike(a);
        out.data = a.data.array() * b.data.array();
        return out;
    }

    /**
     * @brief Multiply a FeatureTensor by a scalar.
     */
    inline FeatureTensor operator*(const FeatureTensor &a, float s)
    {
        FeatureTensor out = makeLike(a);
        out.data = a.data * s;
        return out;
    }

    /**
     * @brief Divide a FeatureTensor by a scalar.
     */
    inline FeatureTensor operator/(const FeatureTensor &a, float s)
    {
        FeatureTensor out = makeLike(a);
        out.data = a.data / s;
        return out;
    }
    /**
     * @brief Add a scalar to every element of a FeatureTensor.
     */
    inline FeatureTensor operator+(const FeatureTensor &a, float s)
    {
        FeatureTensor out = makeLike(a);
        out.data = a.data.array() + s;
        return out;
    }
    /**
     * @brief Subtract a scalar from every element of a FeatureTensor.
     */
    inline FeatureTensor operator-(const FeatureTensor &a, float s)
    {
        FeatureTensor out = makeLike(a);
        out.data = a.data.array() - s;
        return out;
    }

    /// @name Activation and utility functions

    /**
     * @brief Clamp all elements of a FeatureTensor to [0, 1].
     */
    static FeatureTensor clamp01(const FeatureTensor &v)
    {
        FeatureTensor result = makeLike(v);
        result.data = v.data.array().max(0.0f).min(1.0f);

        return result;
    }
    /**
     * @brief Apply sigmoid non-linearity element-wise to a FeatureTensor.
     */
    static FeatureTensor sigmoid(const FeatureTensor &x)
    {
        FeatureTensor result = makeLike(x);
        result.data = 1.0f / (1.0f + (-x.data.array()).exp());

        return result;
    }
    /**
     * @brief Apply tanh non-linearity element-wise to a FeatureTensor.
     */
    static FeatureTensor tanh(const FeatureTensor &x)
    {
        FeatureTensor result = makeLike(x);
        result.data = x.data.array().tanh();

        return result;
    }
    /**
     * @brief Apply ReLU non-linearity element-wise to a FeatureTensor.
     */
    static FeatureTensor relu(const FeatureTensor &x)
    {
        FeatureTensor result = makeLike(x);
        result.data = x.data.array().max(0.0f);

        return result;
    }
    /**
     * @brief Clip all elements of a FeatureTensor to [min_val, max_val].
     */
    static FeatureTensor clip(const FeatureTensor &value, float min_val, float max_val)
    {
        FeatureTensor result = makeLike(value);
        result.data = value.data.array().max(min_val).min(max_val);

        return result;
    }

    /**
     * @brief Abstract base class for feature extractors.
     *
     * Takes an AudioTensor and produces a FeatureTensor.
     */
    class featureExtraction
    {
    public:
        virtual ~featureExtraction() = default;
        /**
         * @brief Extract features from an input audio tensor.
         *
         * @param input Mono or multi-channel audio tensor.
         * @return FeatureTensor containing extracted features.
         */
        virtual FeatureTensor extractFeatures(const AudioTensor &input) = 0;

    protected:
        int numSamples;
        int features;
    };
    /**
     * @brief Abstract base class for differentiable modules.
     *
     * Defines the contract for forward propagation, learning and parameter queries.
     */
    class DifferentiableModule
    {
    public:
        virtual ~DifferentiableModule() = default;

        /**
         * @brief Forward pass through the module.
         *
         * @param input Input features.
         * @return Output features.
         */
        virtual FeatureTensor forward(const FeatureTensor &input) = 0;

        /**
         * @brief Apply a learning step given an error signal.
         *
         * Default implementation is a no-op.
         *
         * @param error Gradient or error tensor.
         */
        virtual void learn(const FeatureTensor &error) {}

        /**
         * @brief Report the number of trainable parameters.
         *
         * @return Number of parameters (default: 0).
         */
        virtual size_t getNumParams() const { return 0; }

        /**
         * @brief Reset any internal state (optional).
         */
        virtual void reset() {}
    };

    // ==============================================================================
    // OPTIMIZER PRIMITIVE
    // ==============================================================================

    /**
     * @brief Abstract optimizer interface for real-time, single-shot updates.
     *
     * Operates on Eigen vectors and matrices.
     */
    class IOptimizer
    {
    public:
        virtual ~IOptimizer() = default;
        /**
         * @brief Update a parameter and return the new value.
         *
         * @param parameter Current parameter vector.
         * @param gradient  Gradient for this parameter.
         * @param features  Current features (for adaptive optimizers).
         * @return Updated parameter vector.
         */
        virtual Eigen::VectorXf update(const Eigen::VectorXf &parameter,
                                       const Eigen::VectorXf &gradient,
                                       const FeatureTensor &features) = 0;
        /// In-place update overload for vector parameters.
        virtual void update(Eigen::VectorXf &parameter,
                            const Eigen::VectorXf &gradient,
                            const FeatureTensor &features) = 0;
        /// In-place update overload for matrix parameters.
        virtual void update(Eigen::MatrixXf &parameter,
                            const Eigen::MatrixXf &gradient,
                            const FeatureTensor &features) = 0;
    };
    /**
     * @brief Base class for real-time Neural DSP layers.
     *
     * Provides block processing, adaptive learning configuration,
     * and an interface for sample-wise processing.
     */
    class NeuralDSPLayer
    {
    public:
        virtual ~NeuralDSPLayer() = default;
        /**
         * @brief Prepare the layer for real-time processing.
         *
         * @param sampleRate Sample rate in Hz.
         */
        virtual void prepare(double sampleRate) = 0; /// prepare for real-time processing
        /**
         * @brief Adapt the layer using an input audio tensor.
         *
         * Called either synchronously or from an async queue.
         */
        virtual void adapt(AudioTensor &input) = 0;

        /**
         * @brief Configure learning behaviour for the layer.
         *
         * @param sr           Sample rate in Hz.
         * @param batchSz      Batch size for adaptation.
         * @param stride       Control stride (in samples).
         * @param asyncEnabled Enable asynchronous adaptation queue.
         */
        void configureLearning(double sr, int batchSz, int stride, bool asyncEnabled)
        {
            sampleRate_ = sr;
            batchSize = batchSz;
            controlStride = stride;
            asyncAdapt = asyncEnabled;
            if (batch_.data.rows() != batchSize || batch_.data.cols() != 1)
            {
                batch_.numSamples = batchSize;
                batch_.features = 1;
                batch_.data.resize(batchSize, 1);
                batch_.data.setZero();
            }
            queue_.resize(queueCapacity);
            for (auto &t : queue_)
            {
                if (t.data.rows() != batchSize || t.data.cols() != 1)
                {
                    t.numSamples = batchSize;
                    t.features = 1;
                    t.data.resize(batchSize, 1);
                }
                t.data.setZero();
            }
            qHead.store(0, std::memory_order_relaxed);
            qTail.store(0, std::memory_order_relaxed);
        }
        /// Set the adaptation batch size (number of samples per update).
        void setBatchSize(int b)
        {
            batchSize = b;
            if (batch_.data.rows() != batchSize || batch_.data.cols() != 1)
            {
                batch_.numSamples = batchSize;
                batch_.features = 1;
                batch_.data.resize(batchSize, 1);
                batch_.data.setZero();
            }
            for (auto &t : queue_)
            {
                if (t.data.rows() != batchSize || t.data.cols() != 1)
                {
                    t.numSamples = batchSize;
                    t.features = 1;
                    t.data.resize(batchSize, 1);
                }
                t.data.setZero();
            }
        }
        /// Set the control stride (in samples).
        void setControlStride(int s) { controlStride = s; }

        /// Enable or disable asynchronous adaptation.
        void setAsyncAdapt(bool e) { asyncAdapt = e; }

        /**
         * @brief Drain any pending adaptation work from the queue.
         *
         * Intended to be called from a non-audio thread.
         */
        void drainAdaptQueue()
        {
            while (qTail.load(std::memory_order_acquire) != qHead.load(std::memory_order_acquire))
            {
                size_t idx = qTail.load(std::memory_order_relaxed);
                // Convert FeatureTensor to AudioTensor before calling adapt
                AudioTensor audio;
                audio.data = queue_[idx].data.col(0); // take first (and only) channel
                audio.numSamples = queue_[idx].numSamples;
                audio.numChannels = 1;
                adapt(audio);
                size_t next = (idx + 1) % queueCapacity;
                qTail.store(next, std::memory_order_release);
            }
        }
        /**
         * @brief Process a single audio sample.
         *
         * @param inputSample Input sample.
         * @return Processed sample.
         */
        virtual float process(float inputSample)
        {
            return inputSample;
        }
        /**
         * @brief Process a block of audio samples.
         *
         * Handles batching and adaptation scheduling.
         *
         * @param input      Input samples.
         * @param output     Output samples.
         * @param numSamples Number of samples to process.
         */
        void processBlock(float *input, float *output, int numSamples)
        {
            assert(batch_.data.rows() == batchSize && batch_.data.cols() == 1);
            if (batchSize == 1)
            {
                batch_.data(0, 0) = input[0];
            }
            else
            {
                int copyCount = std::min(batchSize, numSamples);
                for (int i = 0; i < copyCount; ++i)
                    batch_.data(i, 0) = input[i];
            }
            if (++strideCounter >= controlStride)
            {
                strideCounter = 0;
                if (asyncAdapt)
                {
                    size_t head = qHead.load(std::memory_order_relaxed);
                    size_t next = (head + 1) % queueCapacity;
                    if (next != qTail.load(std::memory_order_acquire))
                    {
                        queue_[head].data = batch_.data;
                        qHead.store(next, std::memory_order_release);
                    }
                }
                else
                {
                    // Convert FeatureTensor to AudioTensor before calling adapt
                    AudioTensor audio;
                    audio.data = batch_.data.col(0); // take first (and only) channel
                    audio.numSamples = batch_.numSamples;
                    audio.numChannels = 1;
                    adapt(audio);
                }
            }

            for (int i = 0; i < numSamples; ++i)
            {
                output[i] = process(input[i]);
            }
        }

    private:
        double sampleRate_;
        int controlStride = 100;
        int strideCounter = 0;
        int batchSize = 1; // single sample batch by default
        FeatureTensor batch_;
        bool asyncAdapt = false;
        static constexpr size_t queueCapacity = 8;
        std::vector<FeatureTensor> queue_;
        std::atomic<size_t> qHead{0};
        std::atomic<size_t> qTail{0};
    };

    /**
     * @brief Deterministic tiny RNG (linear congruential generator).
     */
    static inline uint32_t lcg(uint32_t &s)
    {
        s = 1664525u * s + 1013904223u;
        return s;
    }
    /// Generate a uniform float in [0, 1) using the LCG state.
    static inline float u01(uint32_t &s) { return (lcg(s) >> 8) * (1.0f / 16777216.0f); } // 24-bit
    /// Generate a uniform float in [a, b).
    static inline float randUniform(float a, float b)
    {
        static thread_local uint32_t seed = 0xC0FFEEu;
        return a + (b - a) * u01(seed);
    }

    /**
     * @brief Convert a FeatureTensor error into a single error vector.
     *
     * Uses row 0 if present; otherwise averages all rows.
     */
    static inline Eigen::VectorXf errorVectorRow0(const ChromaFlow::FeatureTensor &err)
    {
        if (err.data.size() == 0)
            return Eigen::VectorXf();
        if (err.data.rows() == 1)
            return err.data.row(0).transpose();
        // if multiple rows, average down to 1 vector
        Eigen::VectorXf v = Eigen::VectorXf::Zero((int)err.data.cols());
        for (Eigen::Index r = 0; r < err.data.rows(); ++r)
            v += err.data.row(r).transpose();
        v /= (float)std::max<Eigen::Index>(1, err.data.rows());
        return v;
    }
    /**
     * @brief Extract row 0 of a FeatureTensor as a column vector.
     */
    static inline Eigen::VectorXf getRowVector(const FeatureTensor &t)
    {
        if (t.data.rows() > 0)
            return t.data.row(0).transpose();

        return Eigen::VectorXf::Zero(t.features);
    }


    /**
     * @brief Safely extract row 0 into a vector of length N.
     *
     * If the tensor has fewer than N columns, the remaining entries are zero.
     */
static inline Eigen::VectorXf getRow0Safe(const FeatureTensor& t, int N)
{
    Eigen::VectorXf v = Eigen::VectorXf::Zero(N);
    
    if (t.data.rows() > 0)
    {
        int T = static_cast<int>(t.data.cols());
        int n = std::min(T, N);
        v.head(n) = t.data.row(0).transpose().head(n);
    }
    
    return v;
}

    /**
     * @brief Wrap an Eigen column vector in a single-row FeatureTensor.
     */

    static inline FeatureTensor toFeatureTensor(const Eigen::VectorXf &v)
    {
        FeatureTensor out;
        out.data.resize(1, v.size());
        out.data.row(0) = v.transpose();
        out.numSamples = 1;
        out.features = (int)v.size();
        return out;
    }

    /// @name Scalar utility functions

    inline float clipf(float v, float minVal, float maxVal)
    {
        return std::max(minVal, std::min(maxVal, v));
    }
    /**
     * @brief Clamp a float value to [0, 1].
     */
    inline float clampf01(float v) { return clipf(v, 0.0f, 1.0f); }

    /**
     * @brief Map milliseconds from [minMs, maxMs] to a curved unit range.
     */
    inline float mapMsToUnit(float ms, float minMs, float maxMs, float curve = 1.0f)
    {
        ms = std::clamp(ms, minMs, maxMs);
        float norm = (ms - minMs) / (maxMs - minMs);
        return std::pow(norm, 1.0f / curve);
    }
    /**
     * @brief Map a curved unit value back to milliseconds in [minMs, maxMs].
     */
    inline float mapUnitToMs(float unit, float minMs, float maxMs, float curve = 1.0f)
    {
        unit = std::clamp(unit, 0.0f, 1.0f);
        float shaped = std::pow(unit, curve);
        return minMs + shaped * (maxMs - minMs);
    }
    /**
     * @brief Map a unit value to a logarithmic frequency in [minFreq, maxFreq].
     */
    inline float mapUnitToLogFrequency(float input, float minFreq = 20.0f, float maxFreq = 20000.0f)
    {
        const float logMin = std::log(minFreq);
        const float logMax = std::log(maxFreq);

        // LERP (Linear Interpolation) in log space
        float logOutput = logMin + input * (logMax - logMin);

        // Convert back to linear space (the frequency in Hz)
        return std::exp(logOutput);
    }
    /**
     * @brief Map a frequency in Hz to a unit value in [0, 1] on a log scale.
     */
    inline float mapLogFrequencyToUnit(float freq, float minFreq = 20.0f, float maxFreq = 20000.0f)
    {
        float safeFreq = std::max(freq, minFreq); // Clamp for safety

        // Calculate the log range
        const float logMin = std::log(minFreq);
        const float logMax = std::log(maxFreq);
        const float logRange = logMax - logMin;

        // Normalize: (Log(Current) - Log(Min)) / Range
        if (logRange == 0.0f)
            return 0.0f;
        float normalized = (std::log(safeFreq) - logMin) / logRange;

        // Clamp to 0..1 range
        return clampf01(normalized);
    }

    /**
     * @brief Map a unit value in [0, 1] to a number of samples.
     */
    inline float mapUnitToTimeSamples(float unitValue, float sampleRate, float minMS = 1.0f, float maxMS = 2000.0f)
    {
        float input = std::max(0.0f, std::min(1.0f, unitValue));

        // 1. Map to Linear Millisecond (MS) Range
        float timeMS = minMS + (input * (maxMS - minMS));

        // 2. Convert MS to Samples: Samples = MS * (Fs / 1000)
        return timeMS * (sampleRate / 1000.0f);
    }
    /**
     * @brief Map a sample count to a unit value in [0, 1].
     */
    inline float mapTimeSamplesToUnit(float samples, float sampleRate, float minMS = 1.0f, float maxMS = 2000.0f)
    {
        // Clamp the input samples for safety
        float safeSamples = std::max(0.0f, samples);

        // 1. Convert Samples back to Milliseconds (MS)
        float timeMS = safeSamples / (sampleRate / 1000.0f);

        // 2. Map MS back to Unit Range (0..1)
        const float rangeMS = maxMS - minMS;
        if (rangeMS == 0.0f)
            return 0.0f;

        float normalized = (timeMS - minMS) / rangeMS;

        // Clamp to 0..1 range
        return clampf01(normalized);
    }
    /**
     * @brief Map a unit value in [0, 1] to a linear range [minValue, maxValue].
     */
    inline float mapUnitToLinearRange(float unitValue, float minValue, float maxValue)
    {
        // Clamp the input unit value for safety
        float input = std::max(0.0f, std::min(1.0f, unitValue));

        // Perform the linear mapping: Output = Min + (Input * Range)
        return minValue + (input * (maxValue - minValue));
    }
    /**
     * @brief Map a value in [minValue, maxValue] to a unit value in [0, 1].
     */
    inline float mapLinearRangeToUnit(float value, float minValue, float maxValue)
    {
        const float range = maxValue - minValue;
        if (range == 0.0f)
            return 0.0f;

        // Normalize: (Current - Min) / Range
        float normalized = (value - minValue) / range;

        // Clamp to 0..1 range
        return clampf01(normalized);
    }
    /**
     * @brief Map a unit value in [0, 1] to an amplitude in dB range [minDB, maxDB].
     */
    inline float mapUnitToAmp(float unitValue, float minDB = -60.0f, float maxDB = 0.0f)
    {
        // 1. Map Unit Value to dB Scale (Linear interpolation in dB space)
        float input = std::max(0.0f, std::min(1.0f, unitValue));
        float dbValue = minDB + (input * (maxDB - minDB));

        // 2. Convert dB to Linear Amplitude: pow(10, dB / 20)
        // Use an anti-denormal guard (std::max(minDB, dbValue)) for safety, though
        // the previous clamp should suffice.
        return std::pow(10.0f, dbValue / 20.0f);
    }
    /**
     * @brief Map an amplitude value to a unit value in [0, 1] using a dB range.
     */
    inline float mapAmpToUnit(float amplitude, float minDB = -60.0f, float maxDB = 0.0f)
    {
        // 1. Convert Linear Amplitude to dB: 20 * log10(Amp)
        // Use an anti-denormal guard (1e-6) for safety against log(0)
        float dbValue = 20.0f * std::log10(std::max(amplitude, 1e-6f));

        // 2. Map dB Value back to Unit Scale
        const float dbRange = maxDB - minDB;
        if (dbRange == 0.0f)
            return 0.0f;

        float normalized = (dbValue - minDB) / dbRange;

        // Clamp to 0..1 range
        return clampf01(normalized);
    }

    /// @}

    /**
     * @brief Utility class for analysing parameter counts and memory footprint.
     */
    class NNAnalyzer
    {
    public:
        /**
         * @brief Count total trainable parameters for a dense or conv layer.
         *
         * @param inputSize  Number of input features/neurons.
         * @param outputSize Number of output features/neurons.
         * @param hasBias    Whether a bias vector is present.
         * @return Total number of parameters.
         */
        static size_t countTrainableParameters(size_t inputSize, size_t outputSize, bool hasBias)
        {
            // Weights: inputSize * outputSize
            // Biases: outputSize (if hasBias is true)
            return (inputSize * outputSize) + (hasBias ? outputSize : 0);
        }

        /**
         * @brief Compute memory footprint (parameters + optimizer state).
         *
         * @param totalParameters Total number of parameters.
         * @param isAdam          True for Adam (two state vectors), false for SGD with momentum.
         * @param sampleTypeSize  Size of scalar type in bytes.
         * @return Estimated memory usage in bytes.
         */
        static size_t calculateTotalMemory(size_t totalParameters, bool isAdam, size_t sampleTypeSize = 4)
        {
            // 1. Memory for the parameters themselves (Weights & Biases)
            size_t memoryForParams = totalParameters * sampleTypeSize;

            // 2. Memory for the Optimizer State (Momentum buffers)
            // SGD (1 state vector: momentum) -> 1 * totalParameters
            // Adam (2 state vectors: m and v) -> 2 * totalParameters
            size_t stateVectors = isAdam ? 2 : 1;
            size_t memoryForState = totalParameters * stateVectors * sampleTypeSize;

            // NOTE: This calculation is conservative and excludes feature/gradient buffers.
            return memoryForParams + memoryForState;
        }

        /// Aggregate parameter count across a set of modules.
        static size_t countTrainableParameters(const std::vector<std::shared_ptr<DifferentiableModule>> &modules)
        {
            size_t total = 0;
            for (const auto &m : modules)
            {
                if (m)
                    total += m->getNumParams();
            }
            return total;
        }
        /**
         * @brief Compute memory footprint for a collection of modules.
         *
         * @param modules        Modules to analyse.
         * @param isAdam         True for Adam (two state vectors), false otherwise.
         * @param sampleTypeSize Size of scalar type in bytes.
         * @return Estimated memory usage in bytes.
         */
        static size_t calculateTotalMemoryForModel(const std::vector<std::shared_ptr<DifferentiableModule>> &modules,
                                                   bool isAdam,
                                                   size_t sampleTypeSize = 4)
        {
            const size_t totalParams = countTrainableParameters(modules);
            const size_t stateVectors = isAdam ? 2 : 1;
            return totalParams * sampleTypeSize + totalParams * stateVectors * sampleTypeSize;
        }
    }; // class NNAnalyzer

} // namespace ChromaFlow
