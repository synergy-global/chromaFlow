#pragma once

#include <vector>

#include <memory>
#include <cmath>
#include "Eigen/Dense"
#include <algorithm>
#include <map>
#include <cassert>
#include <atomic>

// TODO: Add documentation for each primitive
namespace ChromaFlow
{

    // Tensor Primitive=
    struct AudioTensor
    { // 1D audio tensor with shape (numSamples, channels)
        Eigen::VectorXf data;
        int numSamples;
        int numChannels;
    };

    // 2D tensor for feature storage with shape (numSamples, features)
    struct FeatureTensor
    {
        Eigen::MatrixXf data;
        int numSamples;
        int features;
    };

    // data tensor for parameter storage with shape (numParams,)
    struct ParamTensor
    {
        std::map<std::string, float> data;
    };

    // data tensor for weight storage with shape (numFeatures, numParams)
    struct WeightTensor
    {
        Eigen::MatrixXf data;
    };

    // SECTION 1: THE FORMAL BASE CLASS

    /**
     * The formal base class for all primitives, preparing for Phase 3.
     */
    class featureExtraction
    {
    public:
        virtual ~featureExtraction() = default;
        virtual FeatureTensor extractFeatures(const AudioTensor &input) = 0;

    protected:
        int numSamples;
        int features;
    };
    class DifferentiableModule
    {
    public:
        virtual ~DifferentiableModule() = default;

        /**
         * The main processing method.
         */
        // Default forward for neural layers: consume features and emit features
        virtual FeatureTensor forward(const FeatureTensor &input) = 0;

        // Learning hook; default no-op
        virtual void learn(const FeatureTensor &error) {}

        // Report the number of trainable parameters for this module (default: none)
        virtual size_t getNumParams() const { return 0; }

        /**
         * Optional method to reset the internal state of a layer.
         */
        virtual void reset() {}
    };

    // ==============================================================================
    // OPTIMIZER PRIMITIVE
    // ==============================================================================

    /**
     * The abstract base class for all real-time, single-shot optimizers.
     * Defines the contract for updating a learnable parameter.
     */
    class IOptimizer
    {
    public:
        virtual ~IOptimizer() = default;
        /**
         * Updates a learnable parameter based on a calculated gradient.
         * @param parameter The learnable parameter to update (e.g., the 'gamma' in a HybridLayer).
         * @param gradient The error gradient for this parameter.
         * @param features The current audio FeatureVector, for use in adaptive optimizers.
         * @return The updated parameter vector.
         */
        virtual Eigen::VectorXf update(const Eigen::VectorXf &parameter,
                                       const Eigen::VectorXf &gradient,
                                       const FeatureTensor &features) = 0;
        // New in-place overloads to match layer usage patterns
        virtual void update(Eigen::VectorXf &parameter,
                            const Eigen::VectorXf &gradient,
                            const FeatureTensor &features) = 0;
        virtual void update(Eigen::MatrixXf &parameter,
                            const Eigen::MatrixXf &gradient,
                            const FeatureTensor &features) = 0;
    };
    /**
     * The abstract base class for all real-time, single-shot optimizers.
     * Defines the contract for updating a learnable parameter.
     */
    class NeuralDSPLayer
    {
    public:
        virtual ~NeuralDSPLayer() = default;
        /**
         * Prepares the layer for real-time processing (dsp stuff).
         * @param sampleRate The sample rate of the audio signal.
         */
        virtual void prepare(double sampleRate) = 0;                 /// prepare for real-time processing
        /**
         * Adapts the layer to the input features.
         * @param input The input FeatureTensor for adaptation.
         * @return The updated FeatureTensor after adaptation.
         */
        virtual void adapt(AudioTensor &input) = 0; // nn goes here
        /**
         * Configures the learning parameters for the layer.
         * @param sr The sample rate of the audio signal.
         * @param batchSz The batch size for learning.
         * @param stride The control stride for learning.
         * @param asyncEnabled Whether to enable asynchronous adaptation.
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
        void setControlStride(int s) { controlStride = s; }
        void setAsyncAdapt(bool e) { asyncAdapt = e; }
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
         * Processes a single sample of audio.
         * @param inputSample The input audio sample.
         * @return The processed audio sample.
         */
        virtual float process(float inputSample)
        {
            return inputSample;
        }
        // PROCESS BLOCK
        /**
         * Processes a block of audio samples.
         * do not override this method
         * @param input The input audio samples.
         * @param output The output audio samples.
         * @param numSamples The number of samples to process.
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
                for (int i = 0; i < batchSize; ++i)
                {
                    batch_.data(i, 0) = input[i];
                }   
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

    // ==============================================================================
    // UTILITY FUNCTIONS
    // ==============================================================================
    // activation functions
    static FeatureTensor clamp01(FeatureTensor v)
    {
        FeatureTensor result;
        result.data = v.data.array().max(0.0f).min(1.0f);
        result.numSamples = v.numSamples;
        result.features = v.features;
        return result;
    }
    static FeatureTensor sigmoid(FeatureTensor x)
    {
        FeatureTensor result;
        result.data = 1.0f / (1.0f + (-x.data.array()).exp());
        result.numSamples = x.numSamples;
        result.features = x.features;
        return result;
    }

    static FeatureTensor tanh(FeatureTensor x)
    {
        FeatureTensor result;
        result.data = x.data.array().tanh();
        result.numSamples = x.numSamples;
        result.features = x.features;
        return result;
    }

    static FeatureTensor relu(FeatureTensor x)
    {
        FeatureTensor result;
        result.data = x.data.array().max(0.0f);
        result.numSamples = x.numSamples;
        result.features = x.features;
        return result;
    }

    static FeatureTensor clip(FeatureTensor value, float min_val, float max_val)
    {
        FeatureTensor result;
        result.data = value.data.array().max(min_val).min(max_val);
        result.numSamples = value.numSamples;
        result.features = value.features;
        return result;
    }

    

static inline Eigen::VectorXf getRowVector(const FeatureTensor& t)
{
    if (t.data.rows() > 0)
        return t.data.row(0).transpose();

    return Eigen::VectorXf::Zero(t.features);
}

static inline FeatureTensor toFeatureTensor(const Eigen::VectorXf& v)
{
    FeatureTensor out;
    out.data.resize(1, v.size());
    out.data.row(0) = v.transpose();
    out.numSamples = 1;
    out.features = (int)v.size();
    return out;
}
 
    // float versions for utility functions (header-only, so inline)
    inline float clipf(float v, float minVal, float maxVal)
    {
        return std::max(minVal, std::min(maxVal, v));
    }

    inline float clampf01(float v) { return clipf(v, 0.0f, 1.0f); }
        
    inline float mapMsToUnit(float ms, float minMs, float maxMs, float curve = 1.0f)
    {
        ms = std::clamp(ms, minMs, maxMs);
        float norm = (ms - minMs) / (maxMs - minMs);
        return std::pow(norm, 1.0f / curve);
    }
    inline float mapUnitToMs(float unit, float minMs, float maxMs, float curve = 1.0f)
    {
        unit = std::clamp(unit, 0.0f, 1.0f);
        float shaped = std::pow(unit, curve);
        return minMs + shaped * (maxMs - minMs);
    }
    inline float mapUnitToLogFrequency(float unitValue, float minFreq = 20.0f, float maxFreq = 20000.0f)
    {
        float input = std::max(0.0f, std::min(1.0f, unitValue));

        // Calculate the log range for the full frequency spectrum
        const float logMin = std::log(minFreq);
        const float logMax = std::log(maxFreq);

        // LERP (Linear Interpolation) in log space
        float logOutput = logMin + input * (logMax - logMin);

        // Convert back to linear space (the frequency in Hz)
        return std::exp(logOutput);
    }
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



    inline float mapUnitToTimeSamples(float unitValue, float sampleRate, float minMS = 1.0f, float maxMS = 2000.0f)
    {
        float input = std::max(0.0f, std::min(1.0f, unitValue));

        // 1. Map to Linear Millisecond (MS) Range
        float timeMS = minMS + (input * (maxMS - minMS));

        // 2. Convert MS to Samples: Samples = MS * (Fs / 1000)
        return timeMS * (sampleRate / 1000.0f);
    }
    inline float mapTimeSamplesToUnit(float samples, float sampleRate, float minMS = 1.0f, float maxMS = 2000.0f)
    {
        // 1. Convert Samples back to Milliseconds (MS)
        float timeMS = samples / (sampleRate / 1000.0f);

        // 2. Map MS back to Unit Range (0..1)
        const float rangeMS = maxMS - minMS;

        if (rangeMS == 0.0f)
            return 0.0f;

        float normalized = (timeMS - minMS) / rangeMS;

        // Clamp to 0..1 range
        return clampf01(normalized);
    }
    inline float mapUnitToLinearRange(float unitValue, float minValue, float maxValue)
    {
        // Clamp the input unit value for safety
        float input = std::max(0.0f, std::min(1.0f, unitValue));

        // Perform the linear mapping: Output = Min + (Input * Range)
        return minValue + (input * (maxValue - minValue));
    }
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

    /**
     * @brief Utility class for analyzing the size and memory footprint of
     * the Neural Network architecture.
     */
    class NNAnalyzer
    {
    public:
        /**
         * @brief Counts the total number of trainable parameters (weights and biases)
         * in a standard Dense or Conv layer setup.
         * @param inputSize Number of input features/neurons.
         * @param outputSize Number of output features/neurons.
         * @param hasBias Flag if a bias vector is included.
         * @return size_t Total number of trainable parameters (floats).
         */
        static size_t countTrainableParameters(size_t inputSize, size_t outputSize, bool hasBias)
        {
            // Weights: inputSize * outputSize
            // Biases: outputSize (if hasBias is true)
            return (inputSize * outputSize) + (hasBias ? outputSize : 0);
        }

        /**
         * @brief Calculates the total memory footprint of the network in bytes,
         * including the optimizer state.
         * * @param totalParameters The sum of all weights and biases (from countTrainableParameters).
         * @param isAdam Flag indicating if the Adam optimizer (which has two state vectors) is used.
         * @param sampleTypeSize Size of the numeric type (e.g., 4 for float, 8 for double).
         * @return size_t Total estimated memory required in bytes.
         * * Rationale:
         * - Base Memory: Total Parameters * Sample Type Size (e.g., float)
         * - Optimizer State (Momentum/RMS):
         * - SGD with Momentum needs 1 state vector (Momentum).
         * - Adam needs 2 state vectors (Momentum 'm' and Variance 'v').
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

        // Aggregate parameter count across a set of modules
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

        // Aggregate memory footprint across a set of modules
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
