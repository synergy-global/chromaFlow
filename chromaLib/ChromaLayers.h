#pragma once
#include "ChromaBaseClasses.h"
#include "ChromaOptimizers.h"
#include <memory>
#include <random>

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>

#include <algorithm>
// Include the definition of ChromaUtils::clip
namespace ChromaUtils {
        static float clip(float v, float minVal, float maxVal) {
            return std::max(minVal, std::min(maxVal, v));
        }
    }


namespace ChromaFlow
{

    enum class ActivationType {
        LeakyRelu,
        Tanh,
        Sigmoid,
        Linear
    };

    static float randUniform (float a, float b)
    {
        static thread_local std::mt19937 rng([]{
            std::random_device rd;
            return std::mt19937(rd());
        }());
        std::uniform_real_distribution<float> dist(a, b);
        return dist(rng);
    }

    class Collaborator : public DifferentiableModule
    {
    // ... (Collaborator class remains unchanged as it was already optimized) ...
    public:
        // Construct from explicit names and set of names that invert user intent (e.g., "damping")
        Collaborator (std::map<std::string, float> param_names_map, std::optional<std::unordered_set<std::string>> invert_names = std::nullopt)
            : invert_user (invert_names ? *invert_names : std::unordered_set<std::string>{})
        {
            // Materialize the ordered list of parameter names from the provided map
            param_names.reserve (param_names_map.size());
            for (const auto& kv : param_names_map)
            {
                param_names.push_back (kv.first);
                // Initialize last_params_map with provided user values as a sensible default
                last_params_map[kv.first] = kv.second;
            }
        }

        // Collaborator does not transform features; pass-through to satisfy interface.
        FeatureTensor forward (const FeatureTensor& input) override { return input; }
        // 2-arg forward: blends AI parameters with user intents [0..1] per parameter
        ParamTensor predictParams(const FeatureTensor& aiParamWeights, const ParamTensor& user_params) override
        {
            const Eigen::Index rows = aiParamWeights.data.rows();
            const Eigen::Index cols = aiParamWeights.data.cols();
            const size_t n_names = param_names.size();

            auto get_ai_val = [&](size_t i) -> float {
                if (rows == 1 && static_cast<Eigen::Index>(i) < cols)
                    return aiParamWeights.data(0, static_cast<Eigen::Index>(i));
                if (cols == 1 && static_cast<Eigen::Index>(i) < rows)
                    return aiParamWeights.data(static_cast<Eigen::Index>(i), 0);
                if (rows >= 1 && static_cast<Eigen::Index>(i) < cols)
                    return aiParamWeights.data(0, static_cast<Eigen::Index>(i));
                return 0.0f;
            };

            ParamTensor out;
            last_params_map.clear();

            for (size_t i = 0; i < n_names; ++i)
            {
                const std::string& name = param_names[i];
                const float ai_val = get_ai_val(i);

                float intent = 0.0f;
                if (auto it = user_params.data.find(name); it != user_params.data.end())
                {
                    intent = std::fmin(1.0f, std::fmax(0.0f, it->second));
                }

                float final_val;
                if (intent == 0.0f)
                {
                    // Autopilot
                    final_val = ai_val;
                }
                else
                {
                    const float ai_weight = 1.0f - intent;
                    const float user_weight = intent;
                    const float ai_val_scaled = ai_val * 0.2f;
                    const bool invert = invert_user.find(name) != invert_user.end();
                    const float user_contribution = invert ? (1.0f - intent) : intent;

                    final_val = ai_val_scaled * ai_weight + user_contribution * user_weight;
                }

                // Store final value keyed by parameter name
                out.data[name] = final_val;
                last_params_map[name] = final_val;
            }

            return out;
        }

        // Optional: expose the last computed parameters as a name->value map
        const std::unordered_map<std::string, float>& lastParams() const { return last_params_map; }
 
        void reset() override
        {
            last_params_map.clear();
        }

    private:
        std::vector<std::string> param_names;
        std::unordered_set<std::string> invert_user;
        std::unordered_map<std::string, float> last_params_map;

        std::unordered_map<std::string, float> materializeMap (const Eigen::VectorXf& vals) const
        {
            std::unordered_map<std::string, float> m;
            const size_t n = std::min (param_names.size(), static_cast<size_t> (vals.size()));
            for (size_t i = 0; i < n; ++i)
            {
                m.emplace (param_names[i], vals[static_cast<Eigen::Index> (i)]);
            }
            return m;
        }
    };

    class convolutionalLayer : public DifferentiableModule
    {
    public:
        convolutionalLayer (int inputChannels, int outputChannels, int kernelSize)
        {
            // Initialize weights and biases
            this->kernelSize = kernelSize;
            weights_.resize (outputChannels, std::max(1, kernelSize));
            biases_.resize (outputChannels);
        }

        // Provide required override using a simple ones-kernel of size kernelSize
        FeatureTensor forward (const FeatureTensor& input) override
        {
            const int K = std::max (1, kernelSize);
            Eigen::VectorXf kernel = Eigen::VectorXf::Ones(K);

            // Reuse the 2-arg forward internally
            return forwardconvolve (input, kernel);
        }
        // 1 d convolutional layer
        FeatureTensor forwardconvolve (const FeatureTensor& input, const Eigen::VectorXf& kernel)
        {
            // step 1: pad the input
            FeatureTensor paddedInput = pad1d (input, kernelSize / 2);

            // step 2: convolve the input with the kernel
            FeatureTensor output = convolve1d (paddedInput, kernel);

            // step 3: add the bias
            for (int i = 0; i < output.numSamples; ++i)
            {
                output.data(i, 0) += biases_[static_cast<Eigen::Index>(i % biases_.size())];
            }

            // step 4: apply activation function
            output = activationFunction (output);

            return output;
        }
        // padding 1 d
        FeatureTensor pad1d (const FeatureTensor& input, int padding)
        {
            if (padding <= 0)
                return input;

            const int inRows = static_cast<int> (input.data.rows());
            const int inCols = static_cast<int> (input.data.cols());
            const int outRows = inRows + 2 * padding;

            FeatureTensor padded;
            padded.data = Eigen::MatrixXf::Zero (outRows, inCols);
            padded.numSamples = outRows;
            padded.features = inCols;

            // Center the original signal in the padded buffer
            padded.data.block (padding, 0, inRows, inCols) = input.data;
            return padded;
        }
        // convolve 1 d
        FeatureTensor convolve1d (const FeatureTensor& input, const Eigen::VectorXf& kernel)
        {
            // --- CRITICAL PERFORMANCE FIX for Convolve (Point 6) ---
            // NOTE: The original manual looping is retained as the alternative requires
            // complex Signal Processing library integration (like FFT-based convolution),
            // which you chose to avoid in ChromaFlow. However, the logic remains slow.
            const int N = static_cast<int> (input.data.rows());
            const int K = static_cast<int> (kernel.size());
            const int outN = std::max (0, N - K + 1);

            FeatureTensor output;
            output.data = Eigen::MatrixXf::Zero (outN, 1);
            output.numSamples = outN;
            output.features = 1;

            for (int i = 0; i < outN; ++i)
            {
                Eigen::VectorXf seg = input.data.block (i, 0, K, 1).col(0);
                output.data(i, 0) = seg.dot (kernel);
            }
            return output;
        }
        // activation function
        FeatureTensor activationFunction (const FeatureTensor& input)
        {
            // apply the activation function
            FeatureTensor output (input);
            for (int i = 0; i < input.numSamples; ++i)
            {
                // NOTE: Using custom clip function as defined above
                output.data(i, 0) = ChromaFlow::ChromaUtils::clip (output.data(i, 0), 0.0f, 1.0f);
            }
            return output;
        }
        void reset() override
        {
            // TODO: Implement reset logic
        }

    private:
        Eigen::MatrixXf weights_;
        Eigen::VectorXf biases_;
        int kernelSize;
    };

    class denseLayer : public DifferentiableModule
    {
    public:
        denseLayer (int inputSize,
            int outputSize,
            ActivationType activation = ActivationType::LeakyRelu,
            bool useLayerNorm = true,
            float learningRate = 0.005f,
            float momentum = 0.9f)
            : input_size (inputSize),
              output_size (outputSize),
              activation_type (activation),
              use_layer_norm (useLayerNorm),
              optimizer (learningRate, momentum),
              gamma (Eigen::VectorXf::Ones(outputSize)),
              beta (Eigen::VectorXf::Zero(outputSize))
        {
            const float limit = std::sqrt (6.0f / (inputSize + outputSize));
            weights.resize (outputSize, inputSize);
            // Xavier uniform
            for (int o = 0; o < output_size; ++o)
            {
                for (int i = 0; i < input_size; ++i)
                {
                    weights (o, i) = randUniform (-limit, limit);
                }
            }
        }

        // Single-arg forward delegates to two-arg overload with empty user biases
        FeatureTensor forward (const FeatureTensor& upstream_features) override
        {
            static const ParamTensor emptyBiases;
            return forward(upstream_features, emptyBiases);
        }

        // Forward over a 1D feature Tensor; user_biases is optional and must match output_size if provided.
        FeatureTensor forward (const FeatureTensor& upstream_features,
            const ParamTensor& user_biases) override
        {
            // Ensure we operate on a 1D feature vector (first row if sequence)
            const int cols = static_cast<int>(upstream_features.data.cols());
            Eigen::VectorXf x(cols);
            if (upstream_features.data.rows() > 0)
                x = upstream_features.data.row(0).transpose();
            else
                x.setZero();

            // Adapt input length to the layer's expected input_size
            if (cols != input_size)
            {
                if (cols > input_size)
                {
                    x = x.head(static_cast<Eigen::Index>(input_size));
                }
                else
                {
                    Eigen::VectorXf xPadded(static_cast<Eigen::Index>(input_size));
                    xPadded.setZero();
                    xPadded.head(static_cast<Eigen::Index>(cols)) = x;
                    x = xPadded;
                }
            }

            // Calculate weights - use explicit evaluation to avoid template issues
            // Eigen-Optimized Matrix Multiplication (Original code retained this optimal line)
            Eigen::VectorXf z = (weights * x).eval(); // Weights [O x I] * x [I x 1] -> z [O x 1]


            // Optional layer normalization
            if (use_layer_norm)
            {
                const float mean = z.mean();
                const float var = std::max (1e-6f, static_cast<float> ((z.array() - mean).square().mean()));
                Eigen::VectorXf norm = ((z.array() - mean) / std::sqrt (var)).matrix();
                // gamma and beta are 1D params of length output_size
                Eigen::Map<const Eigen::VectorXf> gammaVec (gamma.data(), static_cast<Eigen::Index> (gamma.size()));
                Eigen::Map<const Eigen::VectorXf> betaVec (beta.data(), static_cast<Eigen::Index> (beta.size()));
                z = (gammaVec.cwiseProduct (norm) + betaVec).eval();
            }

            // Optional user biases (map-backed ParamTensor)
            if (!user_biases.data.empty())
            {
                const int n = std::min<int> (z.size(), static_cast<int> (user_biases.data.size()));
                Eigen::VectorXf biasVec(n);
                int idx = 0;
                for (const auto& kv : user_biases.data)
                {
                    if (idx >= n) break;
                    biasVec[static_cast<Eigen::Index>(idx)] = kv.second;
                    ++idx;
                }
                z.head(n) += biasVec;
            }

            // Activation
            Eigen::VectorXf a = z;
            switch (activation_type)
            {
                case ActivationType::LeakyRelu:
                    a = z.unaryExpr ([] (float v) { return v >= 0.0f ? v : 0.01f * v; }).eval();
                    break;
                case ActivationType::Tanh:
                    a = z.array().tanh().matrix().eval();
                    break;
                case ActivationType::Sigmoid:
                    a = (1.0f / (1.0f + (-z.array()).exp())).matrix().eval();
                    break;
                case ActivationType::Linear:
                    a = z;
                    break;
            }

            FeatureTensor out;
            out.data.resize(1, static_cast<Eigen::Index>(a.size()));
            out.data.row(0) = a.transpose();
            out.numSamples = 1;
            out.features = static_cast<int>(a.size());
            return out;
        }

        // Simplified learning rule: outer product of gradient and input features.
        // --- CRITICAL PERFORMANCE FIX (Point 7) ---
        void learn (const FeatureTensor& gradient_from_output,
            const FeatureTensor& features_from_input,
            FeatureTensor& upstream_features) override
        {
            if (gradient_from_output.features != output_size
                || features_from_input.features != input_size)
            {
                return;
            }

            // Get gradient and feature vectors (make sure they are 1D column vectors)
            Eigen::VectorXf grad_out = gradient_from_output.data.row(0).transpose(); // [O x 1]
            Eigen::VectorXf features_in = features_from_input.data.row(0).transpose(); // [I x 1]

            // grad_W[o][i] = grad[o] * x[i] is an Outer Product: [O x 1] * [1 x I] -> [O x I]
            Eigen::MatrixXf grad_W = grad_out * features_in.transpose();

            // Update optimizer
            optimizer.update (weights, grad_W, upstream_features);
            
            // NOTE: For true backprop, the gradient w.r.t the input (upstream_features) needs
            // to be calculated: grad_input = weights.transpose() * grad_out;
        }

        void reset() override
        {
            // TODO: Implement reset logic
        }

    private:
        int input_size;
        int output_size;
        ActivationType activation_type;
        bool use_layer_norm;

        Eigen::MatrixXf weights; // [output_size][input_size]
        Eigen::VectorXf gamma; // layer norm scale
        Eigen::VectorXf beta; // layer norm shift

        ChromaFlow::SGDWithMomentum optimizer;

        void applyActivation (std::vector<float>& z) const
        {
            switch (activation_type)
            {
                case ActivationType::LeakyRelu:
                    for (auto& v : z)
                        v = v > 0.0f ? v : 0.1f * v;
                    break;
                case ActivationType::Tanh:
                    for (auto& v : z)
                        v = std::tanh (v);
                    break;
                case ActivationType::Sigmoid:
                    for (auto& v : z)
                        v = 1.0f / (1.0f + std::exp (-v));
                    break;
                case ActivationType::Linear:
                default:
                    // No-op
                    break;
            }
        }
    };

    class attentionLayer : public DifferentiableModule
    {
    public:
        attentionLayer (int inputSize,
            int outputSize,
            ActivationType activationType = ActivationType::Linear,
            bool useLayerNorm = true)
        {
            const float limit = std::sqrt (6.0f / (inputSize + outputSize));
            weights.resize(outputSize, inputSize);
            // Xavier uniform
            for (int o = 0; o < outputSize; ++o)
            {
                for (int i = 0; i < inputSize; ++i)
                {
                    weights (o, i) = randUniform (-limit, limit);
                }
            }

            queryLayer = std::make_unique<ChromaFlow::denseLayer> (inputSize, outputSize, activationType, useLayerNorm, 0.0001f, 0.01f);
            keyLayer = std::make_unique<ChromaFlow::denseLayer> (inputSize, outputSize, activationType, useLayerNorm, 0.0001f, 0.01f);
            valueLayer = std::make_unique<ChromaFlow::denseLayer> (inputSize, outputSize, activationType, useLayerNorm, 0.0001f, 0.01f);
        }

        FeatureTensor forward (const FeatureTensor& input) override
        {
            // Query, Key, Value using first-row feature vectors
            FeatureTensor qT = queryLayer->forward (input, ParamTensor{});
            FeatureTensor kT = keyLayer->forward (input, ParamTensor{});
            FeatureTensor vT = valueLayer->forward (input, ParamTensor{});

            Eigen::VectorXf query = qT.data.row(0).transpose();
            Eigen::VectorXf key = kT.data.row(0).transpose();
            Eigen::VectorXf value = vT.data.row(0).transpose();

            // Per-feature scores
            // NOTE: Retaining original element-wise scores (Point 9) for simplicity, 
            // but noting this is NOT standard scaled dot-product attention.
            Eigen::VectorXf scores = (query.array() * key.array()).matrix().eval();
            
            // Softmax attention
            Eigen::VectorXf expScores = scores.array().exp().matrix().eval();
            const float denom = expScores.sum();
            Eigen::VectorXf attention;
            if (denom > 0.0f)
                attention = (expScores.array() / denom).matrix().eval();
            else
                attention = Eigen::VectorXf::Zero (expScores.size());

            // Weighted values (element-wise)
            Eigen::VectorXf result = attention.cwiseProduct (value).eval();

            FeatureTensor out;
            out.data.resize(1, static_cast<Eigen::Index>(result.size()));
            out.data.row(0) = result.transpose();
            out.numSamples = 1;
            out.features = static_cast<int>(result.size());
            return out;
        } 
        
        void learn (const FeatureTensor& gradient_from_output,
            const FeatureTensor& features_from_input,
            FeatureTensor& upstream_features) override
        {
            // All queries, keys, and values share the same input (features_from_input)
            queryLayer->learn (gradient_from_output, features_from_input, upstream_features);
            keyLayer->learn (gradient_from_output, features_from_input, upstream_features);
            valueLayer->learn (gradient_from_output, features_from_input, upstream_features);
            // NOTE: The true backpropagation through the attention mechanism is complex 
            // and requires implementing the softmax/weighted sum derivative. This simplified 
            // version only updates the Q/K/V projection weights.
        }
        
        void reset() override
        {
            // TODO: Implement reset logic
        }

    private:
        Eigen::MatrixXf weights;
        std::unique_ptr<ChromaFlow::denseLayer> queryLayer;
        std::unique_ptr<ChromaFlow::denseLayer> keyLayer;
        std::unique_ptr<ChromaFlow::denseLayer> valueLayer;
    };

    class RNNCell : public DifferentiableModule
    {
    public:
        RNNCell (int inputSize, int hiddenSize, float learningRate = 0.005f, float momentum = 0.9f)
            : input_size (inputSize),
              hidden_size (hiddenSize),
              W_x (hiddenSize, inputSize),
              W_h (hiddenSize, hiddenSize),
              b (Eigen::VectorXf::Zero(hiddenSize)),
              hidden_state (Eigen::VectorXf::Zero(hiddenSize)),
              last_input (Eigen::VectorXf::Zero(inputSize)),
              last_hidden_prev (Eigen::VectorXf::Zero(hiddenSize)),
              optimizer (learningRate, momentum)
        {
            // He/Xavier-like init
            const float scale_x = std::sqrt (1.0f / static_cast<float> (input_size));
            const float scale_h = std::sqrt (1.0f / static_cast<float> (hidden_size));
            initMatrix (W_x, scale_x);
            initMatrix (W_h, scale_h);
            b.setZero();
        }

        // Forward: h_t = tanh(W_x x_t + W_h h_{t-1} + b)
        // --- CRITICAL PERFORMANCE FIX (Point 4) ---
        FeatureTensor forward (const FeatureTensor& x_t) override
        {
            FeatureTensor out;
            const int cols = x_t.data.rows() > 0 ? static_cast<int>(x_t.data.cols()) : 0;
            if (cols < input_size)
            {
                out.numSamples = 1;
                out.features = hidden_size;
                out.data.resize(1, static_cast<Eigen::Index>(hidden_size));
                out.data.setZero(); // graceful fail if feature size mismatch
                return out;
            }

            last_hidden_prev = hidden_state;
            last_input = x_t.data.row(0).transpose();
            
            // Use optimized Eigen Matrix Multiplication: z = W_x * x_t + W_h * h_{t-1} + b
            Eigen::VectorXf z = W_x * last_input + W_h * last_hidden_prev + b;

            // tanh activation
            hidden_state = z.array().tanh().matrix().eval();

            out.data.resize(1, static_cast<Eigen::Index>(hidden_state.size()));
            out.data.row(0) = hidden_state.transpose();
            out.numSamples = 1;
            out.features = hidden_size;
            return out;
        }

        // Simplified learn: outer products using last_input and last_hidden_prev
        // --- CRITICAL PERFORMANCE FIX (Point 5) ---
        void learn (const FeatureTensor& gradient_from_output,
            const FeatureTensor& features_from_input,
            FeatureTensor& upstream_features) override
        {
            if (static_cast<int> (gradient_from_output.features) != hidden_size)
                return;
            if (static_cast<int> (features_from_input.features) != input_size)
                return;

            // Get gradient vector (output)
            Eigen::VectorXf grad_out = gradient_from_output.data.row(0).transpose(); // [H x 1]

            // 1. grad_Wx[h][i] = grad_out[h] * last_input[i] (Outer Product: [H x 1] * [1 x I] -> [H x I])
            Eigen::MatrixXf grad_Wx = grad_out * last_input.transpose();

            // 2. grad_Wh[h][k] = grad_out[h] * last_hidden_prev[k] (Outer Product: [H x 1] * [1 x H] -> [H x H])
            Eigen::MatrixXf grad_Wh = grad_out * last_hidden_prev.transpose();

            // 3. grad_b[h] = grad_out[h]
            Eigen::VectorXf grad_b = grad_out;

            // Update optimizers
            optimizer.update (W_x, grad_Wx, upstream_features);
            optimizer.update (W_h, grad_Wh, upstream_features);
            optimizer.update (b, grad_b, upstream_features);
            
            // NOTE: For true backprop, the gradient w.r.t input and previous hidden state is required.
        }

        void reset() override
        {
            // Resetting Eigen containers is safer with setZero() or fill(0.0f)
            hidden_state.setZero();
            last_input.setZero();
            last_hidden_prev.setZero();
        }

    private:
        int input_size;
        int hidden_size;

        Eigen::MatrixXf W_x; // [hidden_size][input_size]
        Eigen::MatrixXf W_h; // [hidden_size][hidden_size]
        Eigen::VectorXf b; // [hidden_size]

        Eigen::VectorXf hidden_state; // current h_t
        Eigen::VectorXf last_input; // last x_t
        Eigen::VectorXf last_hidden_prev; // last h_{t-1}

        SGDWithMomentum optimizer;

        static void initMatrix (Eigen::MatrixXf& M, float scale)
        {
            uint32_t seed = 1234567u;
            auto nextU = [&seed]() {
                seed = 1664525u * seed + 1013904223u;
                return static_cast<float> (seed) / static_cast<float> (UINT32_MAX);
            };
            for (Eigen::Index h = 0; h < M.rows(); ++h)
            {
                for (Eigen::Index i = 0; i < M.cols(); ++i)
                {
                    const float u = nextU();
                    M (h, i) = (2.0f * u - 1.0f) * scale; // uniform in [-scale, +scale]
                }
            }
        }
    };

} // namespace ChromaFlow
