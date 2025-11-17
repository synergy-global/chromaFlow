#pragma once

#include <vector>

#include <memory>
#include <cmath>
#include "Eigen/Dense"
#include <algorithm>
#include <map>
namespace ChromaFlow
{

    // ==============================================================================
    // Tensor Primitive==============================================================================
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
        // Optional forward overload with user biases (dense-like); default delegates to single-arg
        virtual FeatureTensor forward(const FeatureTensor &upstream_features,
                                      const ParamTensor &user_biases) { return forward(upstream_features); }

        // The extractor is the only module consuming AudioTensor directly
        // virtual FeatureTensor extractFeatures(const AudioTensor& input) = 0;
        // Collaborator blends AI weights with user params to emit final parameters
        virtual ParamTensor predictParams(const FeatureTensor &aiParamWeights, const ParamTensor &user_params)
        {
            // Default: modules that don't produce parameters just pass through user params.
            return user_params;
        }
        // Learning hook; default no-op
        virtual void learn(const FeatureTensor &gradient_from_output,
                           const FeatureTensor &features_from_input,
                           FeatureTensor &upstream_features) {}

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

    // ==============================================================================
    // UTILITY FUNCTIONS
    // ==============================================================================

    class ChromaUtils
    {
    public:
        // Local clamp for 0..1 range to avoid relying on std::clamp (C++17)
        static float clamp01(float v)
        {
            return std::max(0.0f, std::min(1.0f, v));
        }
        static float sigmoid(float x)
        {
            return 1.0f / (1.0f + std::exp(-x));
        }

        static float tanh(float x)
        {
            return std::tanh(x);
        }

        static float relu(float x)
        {
            return std::max(0.0f, x);
        }

        static float clip(float value, float min_val, float max_val)
        {
            return std::max(min_val, std::min(max_val, value));
        }

        static std::vector<float> clipVector(const std::vector<float> &vec, float min_val, float max_val)
        {
            std::vector<float> result(vec.size());
            for (size_t i = 0; i < vec.size(); ++i)
            {
                result[i] = clip(vec[i], min_val, max_val);
            }
            return result;
        }

        static float calculateRMS(const std::vector<float> &audioBlock)
        {
            if (audioBlock.empty())
                return 0.0f;
            float sum = 0.0f;
            for (float sample : audioBlock)
            {
                sum += sample * sample;
            }
            return std::sqrt(sum / audioBlock.size());
        }

        static float calculateMeanAbs(const std::vector<float> &audioBlock)
        {
            if (audioBlock.empty())
                return 0.0f;
            float sum = 0.0f;
            for (float sample : audioBlock)
            {
                sum += std::abs(sample);
            }
            return sum / audioBlock.size();
        }

        static float calculateStd(const std::vector<float> &audioBlock)
        {
            if (audioBlock.empty())
                return 0.0f;
            float mean = 0.0f;
            for (float sample : audioBlock)
            {
                mean += sample;
            }
            mean /= audioBlock.size();

            float variance = 0.0f;
            for (float sample : audioBlock)
            {
                float diff = sample - mean;
                variance += diff * diff;
            }
            variance /= audioBlock.size();
            return std::sqrt(variance);
        }

        static float calculateSkewness(const std::vector<float> &data)
        {
            if (data.size() < 3)
                return 0.0f;

            float mean = 0.0f;
            for (float val : data)
                mean += val;
            mean /= data.size();

            float m2 = 0.0f, m3 = 0.0f;
            for (float val : data)
            {
                float diff = val - mean;
                m2 += diff * diff;
                m3 += diff * diff * diff;
            }
            m2 /= data.size();
            m3 /= data.size();

            float std_dev = std::sqrt(m2);
            if (std_dev < 1e-8f)
                return 0.0f;

            return m3 / (std_dev * std_dev * std_dev);
        }

        static float calculateKurtosis(const std::vector<float> &data)
        {
            if (data.size() < 4)
                return 0.0f;

            float mean = 0.0f;
            for (float val : data)
                mean += val;
            mean /= data.size();

            float m2 = 0.0f, m4 = 0.0f;
            for (float val : data)
            {
                float diff = val - mean;
                float diff2 = diff * diff;
                m2 += diff2;
                m4 += diff2 * diff2;
            }
            m2 /= data.size();
            m4 /= data.size();

            if (m2 < 1e-8f)
                return 0.0f;

            return m4 / (m2 * m2) - 3.0f;
        }

        float mapUnitToLogFrequency(float unitValue, float minFreq = 20.0f, float maxFreq = 20000.0f)
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
        float mapLogFrequencyToUnit(float freq, float minFreq = 20.0f, float maxFreq = 20000.0f)
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
            return ChromaUtils::clamp01(normalized);
        }

        float mapUnitToTimeSamples(float unitValue, float sampleRate, float minMS = 1.0f, float maxMS = 2000.0f)
        {
            float input = std::max(0.0f, std::min(1.0f, unitValue));

            // 1. Map to Linear Millisecond (MS) Range
            float timeMS = minMS + (input * (maxMS - minMS));

            // 2. Convert MS to Samples: Samples = MS * (Fs / 1000)
            return timeMS * (sampleRate / 1000.0f);
        }
        float mapTimeSamplesToUnit(float samples, float sampleRate, float minMS = 1.0f, float maxMS = 2000.0f)
        {
            // 1. Convert Samples back to Milliseconds (MS)
            float timeMS = samples / (sampleRate / 1000.0f);

            // 2. Map MS back to Unit Range (0..1)
            const float rangeMS = maxMS - minMS;

            if (rangeMS == 0.0f)
                return 0.0f;

            float normalized = (timeMS - minMS) / rangeMS;

            // Clamp to 0..1 range
            return ChromaUtils::clamp01(normalized);
        }
        float mapUnitToLinearRange(float unitValue, float minValue, float maxValue)
        {
            // Clamp the input unit value for safety
            float input = std::max(0.0f, std::min(1.0f, unitValue));

            // Perform the linear mapping: Output = Min + (Input * Range)
            return minValue + (input * (maxValue - minValue));
        }
        float mapLinearRangeToUnit(float value, float minValue, float maxValue)
        {
            const float range = maxValue - minValue;
            if (range == 0.0f)
                return 0.0f;

            // Normalize: (Current - Min) / Range
            float normalized = (value - minValue) / range;

            // Clamp to 0..1 range
            return ChromaUtils::clamp01(normalized);
        }
        float mapUnitToAmp(float unitValue, float minDB = -60.0f, float maxDB = 0.0f)
        {
            // 1. Map Unit Value to dB Scale (Linear interpolation in dB space)
            float input = std::max(0.0f, std::min(1.0f, unitValue));
            float dbValue = minDB + (input * (maxDB - minDB));

            // 2. Convert dB to Linear Amplitude: pow(10, dB / 20)
            // Use an anti-denormal guard (std::max(minDB, dbValue)) for safety, though
            // the previous clamp should suffice.
            return std::pow(10.0f, dbValue / 20.0f);
        }
        float mapAmpToUnit(float amplitude, float minDB = -60.0f, float maxDB = 0.0f)
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
            return ChromaUtils::clamp01(normalized);
        }
    };

/**
 * @brief Utility class for analyzing the size and memory footprint of 
 * the Neural Network architecture.
 */
class NNAnalyzer {
public:
    /**
     * @brief Counts the total number of trainable parameters (weights and biases) 
     * in a standard Dense or Conv layer setup.
     * @param inputSize Number of input features/neurons.
     * @param outputSize Number of output features/neurons.
     * @param hasBias Flag if a bias vector is included.
     * @return size_t Total number of trainable parameters (floats).
     */
    static size_t countTrainableParameters(size_t inputSize, size_t outputSize, bool hasBias) {
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
    static size_t calculateTotalMemory(size_t totalParameters, bool isAdam, size_t sampleTypeSize = 4) {
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
    static size_t countTrainableParameters(const std::vector<std::shared_ptr<DifferentiableModule>>& modules) {
        size_t total = 0;
        for (const auto& m : modules) {
            if (m) total += m->getNumParams();
        }
        return total;
    }

    // Aggregate memory footprint across a set of modules
    static size_t calculateTotalMemoryForModel(const std::vector<std::shared_ptr<DifferentiableModule>>& modules,
                                               bool isAdam,
                                               size_t sampleTypeSize = 4) {
        const size_t totalParams = countTrainableParameters(modules);
        const size_t stateVectors = isAdam ? 2 : 1;
        return totalParams * sampleTypeSize + totalParams * stateVectors * sampleTypeSize;
    }
};

} // namespace ChromaFlow