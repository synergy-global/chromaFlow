#pragma once
#include "ChromaBaseClasses.h"
#include "ChromaOptimizers.h"
#include <memory>
#include <random>

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include <cassert>
#include <map>
#include <cmath>
#include <cstdint>

#ifndef jassert
#define jassert(x) assert(x)
#define jassertfalse assert(false)
#endif

// ============================
// Agent mode (forward-only)
// ============================
#ifndef CHROMAFLOW_AGENT_MODE
#define CHROMAFLOW_AGENT_MODE 1
#endif

namespace ChromaUtils
{

    static inline float clip(float v, float minVal, float maxVal)
    {
        return std::max(minVal, std::min(maxVal, v));
    }
    static inline float clip01(float v) { return clip(v, 0.0f, 1.0f); }

    // Deterministic tiny RNG (no random_device)
    static inline uint32_t lcg(uint32_t &s)
    {
        s = 1664525u * s + 1013904223u;
        return s;
    }
    static inline float u01(uint32_t &s) { return (lcg(s) >> 8) * (1.0f / 16777216.0f); } // 24-bit
    static inline float randUniform(float a, float b)
    {
        static thread_local uint32_t seed = 0xC0FFEEu;
        return a + (b - a) * u01(seed);
    }

    // Stable 1st-order adaptation primitive (DSP-ish)
    struct IIRAdapt
    {
        float mu = 1e-4f;
        float minV = -2.0f;
        float maxV = 2.0f;

        inline float step(float &p, float target)
        {
            p += mu * (target - p);
            p = clip(p, minV, maxV);
            return p;
        }

        inline void stepVec(Eigen::VectorXf &p, const Eigen::VectorXf &target)
        {
            const int n = std::min<int>(p.size(), target.size());
            for (int i = 0; i < n; ++i)
                step(p[i], target[i]);
        }
    };

    // Streaming stats (bounded + stable)
    struct RunningStats
    {
        float a = 0.01f;
        float mean = 0.0f;
        float var = 1.0f;

        inline void reset(float mean0 = 0.0f, float var0 = 1.0f)
        {
            mean = mean0;
            var = var0;
        }

        inline void push(float x)
        {
            const float d = x - mean;
            mean += a * d;
            var += a * (d * d - var);
            var = std::max(var, 1e-8f);
        }
    };

} // namespace ChromaUtils

namespace ChromaFlow
{

    enum class ActivationType
    {
        LeakyRelu,
        Tanh,
        Sigmoid,
        Linear
    };

    class Collaborator : public DifferentiableModule
    {
    public:
        Collaborator(std::map<std::string, float> param_names_map,
                     std::optional<std::unordered_set<std::string>> invert_names = std::nullopt)
            : invert_user(invert_names.value_or(std::unordered_set<std::string>{}))
        {
            param_names.reserve(param_names_map.size());
            for (const auto &kv : param_names_map)
            {
                param_names.push_back(kv.first);
                last_params_map[kv.first] = kv.second;
            }
        }

        FeatureTensor forward(const FeatureTensor &input) override { return input; }

        ParamTensor predictParams(const FeatureTensor &aiParamWeights, const ParamTensor &user_params) override
        {
            const Eigen::Index rows = aiParamWeights.data.rows();
            const Eigen::Index cols = aiParamWeights.data.cols();
            const size_t n_names = param_names.size();

            auto get_ai_val = [&](size_t i) -> float
            {
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
                const std::string &name = param_names[i];
                const float ai_val = get_ai_val(i);

                float intent = 0.0f;
                auto it = user_params.data.find(name);
                if (it != user_params.data.end())
                    intent = std::fmin(1.0f, std::fmax(0.0f, it->second));

                float final_val;
                if (intent == 0.0f)
                {
                    final_val = ai_val;
                }
                else
                {
                    const float ai_weight = 1.0f - intent;
                    const float user_weight = intent;
                    const float ai_val_scaled = ai_val * 0.2f;
                    const bool invert = invert_user.find(name) != invert_user.end();
                    const float user_contribution = invert ? (1.0f - it->second) : it->second;

                    final_val = ai_val_scaled * ai_weight + user_contribution * user_weight;
                }

                out.data[name] = final_val;
                last_params_map[name] = final_val;
            }

            return out;
        }

        const std::unordered_map<std::string, float> &lastParams() const { return last_params_map; }

        void reset() override { last_params_map.clear(); }

    private:
        std::vector<std::string> param_names;
        std::unordered_set<std::string> invert_user;
        std::unordered_map<std::string, float> last_params_map;
    };

    // ============================================================================
    // convolutionalLayer (1D)
    // Agent-mode: adapt only small affine (gain/bias), keep kernel stable.
    // ============================================================================
    class convolutionalLayer : public DifferentiableModule
    {
    public:
        convolutionalLayer(int inputChannels, int outputChannels, int kernelSize)
        {
            (void)inputChannels;
            (void)outputChannels;
            kernelSize_ = std::max(1, kernelSize);
            kernel_.setZero(kernelSize_);
            kernel_(kernelSize_ / 2) = 1.0f; // identity-ish
            biases_.setZero(1);

            // agent coefficients
            gain_ = 1.0f;
            bias_affine_ = 0.0f;
            inStats_.reset(0.0f, 1.0f);
            outStats_.reset(0.0f, 1.0f);
        }

        FeatureTensor forward(const FeatureTensor &input) override
        {
            last_input = input.data;
            const int K = kernelSize_;
            Eigen::VectorXf kernel = kernel_.size() == K ? kernel_ : Eigen::VectorXf::Ones(K);
            return forwardconvolve(input, kernel);
        }

        FeatureTensor forwardconvolve(const FeatureTensor &input, const Eigen::VectorXf &kernel)
        {
            const int K = static_cast<int>(kernel.size());
            if (K <= 0)
                return input;

            const int pad = K / 2;
            const int inN = static_cast<int>(input.data.rows());
            const int outN = std::max(0, inN - K + 1) + 2 * pad;

            FeatureTensor output;
            output.data = Eigen::MatrixXf::Zero(outN, 1);
            output.numSamples = outN;
            output.features = 1;

            Eigen::VectorXf col = input.data.col(0);
            const float *dataPtr = col.data();

            // stats (input)
            for (int i = 0; i < inN; ++i)
                inStats_.push(dataPtr[i]);

            Eigen::VectorXf kr = kernel.reverse();
            float *outPtr = output.data.data();

            const int validOut = std::max(0, inN - K + 1);
            for (int i = 0; i < validOut; ++i)
            {
                Eigen::Map<const Eigen::VectorXf> segMap(dataPtr + i, K);
                float acc = segMap.dot(kr);
                outPtr[i + pad] = acc;
            }

            const float b0 = biases_.size() ? biases_(0) : 0.0f;
            for (int i = 0; i < outN; ++i)
            {
                float v = outPtr[i] + b0;

                // agent affine
                v = v * gain_ + bias_affine_;
                outPtr[i] = v; // simplest
                               // or: outPtr[i] = ChromaUtils::clip(v, -4.0f, 4.0f);

                outStats_.push(outPtr[i]);
            }

            last_output = output.data;
            return output;
        }

        // Backprop retained (debug/tests). Not used in Agent Mode.
        std::pair<FeatureTensor, FeatureTensor> backward(const FeatureTensor &grad_out)
        {
            FeatureTensor grad_prev;
            FeatureTensor act_prev;

            if (grad_out.data.size() == 0)
            {
                grad_prev.data = Eigen::MatrixXf::Zero(last_input.rows(), last_input.cols());
                grad_prev.numSamples = last_input.rows();
                grad_prev.features = last_input.cols();
                act_prev.data = last_input;
                act_prev.numSamples = last_input.rows();
                act_prev.features = last_input.cols();
                return {grad_prev, act_prev};
            }

            // FIX: read gradient as col OR row (outN×1 is typical)
            Eigen::VectorXf g;
            if (grad_out.data.rows() == 1)
                g = grad_out.data.row(0).transpose();
            else
                g = grad_out.data.col(0);

            const int K = kernelSize_;
            const int inN = static_cast<int>(last_input.rows());
            const int outN = static_cast<int>(g.size());
            const int pad = K / 2;
            const int validOut = std::max(0, inN - K + 1);

            Eigen::VectorXf gradKernel = Eigen::VectorXf::Zero(K);
            Eigen::VectorXf inputCol = last_input.col(0);
            const float *dataPtr = inputCol.data();
            for (int i = 0; i < validOut; ++i)
            {
                float go = g[i + pad];
                if (go == 0.f)
                    continue;
                Eigen::Map<const Eigen::VectorXf> segMap(dataPtr + i, K);
                gradKernel += go * segMap.reverse();
            }

            Eigen::VectorXf gradInputVec = Eigen::VectorXf::Zero(inN);
            Eigen::VectorXf kr = kernel_.reverse();
            for (int i = 0; i < validOut; ++i)
            {
                float go = g[i + pad];
                if (go == 0.f)
                    continue;
                for (int k = 0; k < K; ++k)
                    gradInputVec[i + k] += go * kr(k);
            }

            grad_prev.data = Eigen::MatrixXf::Zero(inN, 1);
            grad_prev.data.col(0) = gradInputVec;
            grad_prev.numSamples = inN;
            grad_prev.features = 1;

            act_prev.data = last_input;
            act_prev.numSamples = last_input.rows();
            act_prev.features = last_input.cols();

#if !CHROMAFLOW_AGENT_MODE
            // classic optimizer update
            Eigen::VectorXf negGrad = -gradKernel;
            optimizer.update(kernel_, negGrad, grad_prev);

            Eigen::VectorXf gradB(1);
            gradB(0) = g.sum();
            optimizer.update(biases_, -gradB, grad_prev);
#else
            (void)outN;
#endif
            return {grad_prev, act_prev};
        }

        // Agent learning: explicit feedback = slow adaptation of gain/bias only
        void learn(const FeatureTensor &gradient_from_output,
                   const FeatureTensor &features_from_input,
                   FeatureTensor &upstream_features) override
        {
            (void)gradient_from_output;
            (void)features_from_input;
            (void)upstream_features;

            // Homeostasis: match output mean to input mean (both typically in [0,1])
            const float targetMean = ChromaUtils::clip(inStats_.mean, 0.0f, 1.0f);

            // Bias corrects mean
            adapt_.mu = 1e-4f;
            adapt_.minV = -1.0f;
            adapt_.maxV = 1.0f;
            adapt_.step(bias_affine_, bias_affine_ + (targetMean - outStats_.mean));

            // Gain corrects variance (match std)
            const float inStd = std::sqrt(inStats_.var);
            const float outStd = std::sqrt(outStats_.var);
            const float ratio = (inStd + 1e-6f) / (outStd + 1e-6f);

            adapt_.mu = 5e-5f;
            adapt_.minV = 0.05f;
            adapt_.maxV = 5.0f;
            adapt_.step(gain_, gain_ * ratio);
        }

        void setKernel(const Eigen::VectorXf &k)
        {
            kernel_ = k;
            kernelSize_ = static_cast<int>(k.size());
        }
        void setBiases(const Eigen::VectorXf &b) { biases_ = b; }
        const Eigen::VectorXf &getWeights() const { return kernel_; }
        const Eigen::VectorXf &getBiases() const { return biases_; }

        void reset() override
        {
            last_input.setZero();
            last_output.setZero();
            inStats_.reset(0.0f, 1.0f);
            outStats_.reset(0.0f, 1.0f);
            gain_ = 1.0f;
            bias_affine_ = 0.0f;
        }

        size_t getNumParams() const override
        {
            // kernel + biases + agent affine
            return static_cast<size_t>(kernel_.size()) + static_cast<size_t>(biases_.size()) + 2u;
        }

    private:
        Eigen::VectorXf kernel_;
        Eigen::VectorXf biases_;
        int kernelSize_ = 1;

        Eigen::MatrixXf last_input;
        Eigen::VectorXf last_output;

        // agent coeffs + stats
        float gain_ = 1.0f;
        float bias_affine_ = 0.0f;
        ChromaUtils::IIRAdapt adapt_;
        ChromaUtils::RunningStats inStats_;
        ChromaUtils::RunningStats outStats_;

        SGDWithMomentum optimizer;
    };

    // ============================================================================
    // denseLayer
    // Agent-mode: freeze weights, adapt only gamma/beta (+ alpha optionally but slow)
    // ============================================================================
    class denseLayer : public DifferentiableModule
    {
    public:
        denseLayer(int inputSize,
                   int outputSize,
                   ActivationType activation = ActivationType::LeakyRelu,
                   bool useLayerNorm = true,
                   float learningRate = 0.005f,
                   float momentum = 0.9f)
            : input_size(inputSize),
              output_size(outputSize),
              activation_type(activation),
              use_layer_norm(useLayerNorm),
              optimizer(learningRate, momentum),
              optimizer_alpha(learningRate, momentum),
              gamma(Eigen::VectorXf::Ones(outputSize)),
              beta(Eigen::VectorXf::Zero(outputSize))
        {
            const float limit = std::sqrt(6.0f / (inputSize + outputSize));
            weights.resize(outputSize, inputSize);
            for (int o = 0; o < output_size; ++o)
                for (int i = 0; i < input_size; ++i)
                    weights(o, i) = ChromaUtils::randUniform(-limit, limit);

            input_ema_state.setZero(input_size);
            learnable_alpha.setConstant(input_size, 0.5f);

            outMean_.reset(0.0f, 1.0f);
            outVar_.reset(0.0f, 1.0f);
        }

        FeatureTensor forward(const FeatureTensor &upstream_features) override
        {
            const int cols = static_cast<int>(upstream_features.data.cols());
            Eigen::VectorXf x(input_size);
            x.setZero();

            if (upstream_features.data.rows() > 0)
            {
                int available = std::min(cols, input_size);
                x.head(available) = upstream_features.data.row(0).transpose().head(available);
                last_external_input = upstream_features.data.row(0).transpose();
            }

            for (int i = 0; i < learnable_alpha.size(); ++i)
                learnable_alpha[i] = ChromaUtils::clip(learnable_alpha[i], 0.001f, 0.999f);

            Eigen::VectorXf one_minus_alpha = Eigen::VectorXf::Constant(input_size, 1.0f) - learnable_alpha;
            input_ema_state = input_ema_state.cwiseProduct(one_minus_alpha) + x.cwiseProduct(learnable_alpha);

            last_input_ema = input_ema_state;

            Eigen::VectorXf z = (weights * input_ema_state).eval();
            last_preact = z;

            if (use_layer_norm)
            {
                const float mean = z.mean();
                const float var = std::max(1e-6f, static_cast<float>((z.array() - mean).square().mean()));
                Eigen::VectorXf norm = ((z.array() - mean) / std::sqrt(var)).matrix();
                last_xhat = norm;
                z = (gamma.cwiseProduct(norm) + beta).eval();
            }
            else
            {
                // always apply beta even if LN disabled (acts as bias)
                z = z + beta;
            }

            last_postnorm = z;

            Eigen::VectorXf a = z.unaryExpr([this](float v)
                                            {
            switch (activation_type) {
                case ActivationType::LeakyRelu: return v >= 0.0f ? v : 0.01f * v;
                case ActivationType::Tanh: return std::tanh (v);
                case ActivationType::Sigmoid: return 1.0f / (1.0f + std::exp (-v));
                case ActivationType::Linear: default: return v;
            } })
                                    .eval();

            last_activation = a;

            // stats for agent learning
            {
                const float m = a.mean();
                const float vv = static_cast<float>((a.array() - m).square().mean());
                outMean_.push(m);
                outVar_.push(vv);
            }

            FeatureTensor out;
            out.data.resize(1, static_cast<Eigen::Index>(a.size()));
            out.data.row(0) = a.transpose();
            out.numSamples = 1;
            out.features = static_cast<int>(a.size());
            return out;
        }

        // Backprop retained (debug/tests). Not used for production agent learning.
        std::pair<FeatureTensor, FeatureTensor> backward(const FeatureTensor &grad_output)
        {
            FeatureTensor grad_prev;
            FeatureTensor act_prev;

            if (grad_output.data.size() == 0)
            {
                grad_prev.data = Eigen::MatrixXf::Zero(last_input_ema.size(), 1);
                grad_prev.numSamples = static_cast<int>(last_input_ema.size());
                grad_prev.features = 1;

                act_prev.data = last_input_ema;
                act_prev.numSamples = static_cast<int>(last_input_ema.size());
                act_prev.features = 1;
                return {grad_prev, act_prev};
            }

            const int goRows = grad_output.data.rows();
            const int goCols = grad_output.data.cols();
            jassert(goRows == 1 || goCols == 1);

            Eigen::VectorXf grad;
            if (goRows == 1)
                grad = grad_output.data.row(0).transpose().eval();
            else
                grad = grad_output.data.col(0).eval();

            const int outDim = static_cast<int>(grad.size());

            jassert(last_activation.size() == outDim);
            jassert(last_postnorm.size() == outDim);
            jassert(last_preact.size() == outDim);
            if (use_layer_norm)
            {
                jassert(last_xhat.size() == outDim);
                jassert(gamma.size() == outDim);
            }
            jassert(beta.size() == outDim);

            Eigen::VectorXf dA(outDim);
            for (int i = 0; i < outDim; ++i)
            {
                float v = last_postnorm(i);
                switch (activation_type)
                {
                case ActivationType::LeakyRelu:
                    dA(i) = v >= 0.0f ? 1.0f : 0.01f;
                    break;
                case ActivationType::Tanh:
                    dA(i) = 1.0f - std::tanh(v) * std::tanh(v);
                    break;
                case ActivationType::Sigmoid:
                {
                    float s = 1.0f / (1.0f + std::exp(-v));
                    dA(i) = s * (1.0f - s);
                    break;
                }
                case ActivationType::Linear:
                default:
                    dA(i) = 1.0f;
                    break;
                }
            }

            Eigen::VectorXf dPre = grad.cwiseProduct(dA).eval();

            Eigen::VectorXf dgamma = Eigen::VectorXf::Zero(gamma.size());
            Eigen::VectorXf dbeta = dPre;
            Eigen::VectorXf dZ = dPre;

            if (use_layer_norm)
            {
                dgamma = dPre.cwiseProduct(last_xhat);

                const float mean = last_preact.mean();
                const float var = std::max(1e-6f,
                                           static_cast<float>((last_preact.array() - mean)
                                                                  .square()
                                                                  .mean()));
                const float invStd = 1.0f / std::sqrt(var);
                const float Nf = static_cast<float>(last_preact.size());

                Eigen::VectorXf xmu = last_preact.array() - mean;
                Eigen::VectorXf dxhat = dPre.cwiseProduct(gamma);

                const float sum_dxhat = dxhat.sum();
                const float sum_dxhat_xmu = (dxhat.cwiseProduct(xmu)).sum();

                Eigen::VectorXf dz =
                    (dxhat.array() - sum_dxhat / Nf - xmu.array() * (sum_dxhat_xmu / (var * Nf))).matrix() * invStd;

                dZ = dz;
            }

            const int inDim = static_cast<int>(last_input_ema.size());
            jassert(weights.rows() == outDim);
            jassert(weights.cols() == inDim);

            Eigen::MatrixXf dW = dZ * last_input_ema.transpose();
            Eigen::VectorXf dInput = weights.transpose() * dZ;

            grad_prev.data = Eigen::MatrixXf::Zero(inDim, 1);
            grad_prev.data.col(0) = dInput;
            grad_prev.numSamples = static_cast<int>(dInput.size());
            grad_prev.features = 1;

            act_prev.data = last_input_ema;
            act_prev.numSamples = static_cast<int>(last_input_ema.size());
            act_prev.features = 1;

#if !CHROMAFLOW_AGENT_MODE
            optimizer.update(weights, -dW, grad_output);
            if (gamma.size() == outDim)
                optimizer.update(gamma, -dgamma, grad_output);
            optimizer.update(beta, -dbeta, grad_output);

            if (last_external_input.size() == learnable_alpha.size() && learnable_alpha.size() == inDim)
            {
                Eigen::VectorXf error_signal_mu = last_external_input - last_input_ema;

                Eigen::VectorXf d_alpha_raw =
                    (weights.transpose() * dZ)
                        .cwiseProduct(error_signal_mu)
                        .cwiseProduct(learnable_alpha
                                          .cwiseProduct(Eigen::VectorXf::Ones(learnable_alpha.size()) - learnable_alpha));

                optimizer_alpha.update(learnable_alpha, -d_alpha_raw, grad_output);
            }
#endif

            return {grad_prev, act_prev};
        }

        // Agent learning: explicit feedback on a *small* set (gamma/beta [+alpha optional])
        void learn(const FeatureTensor &gradient_from_output,
                   const FeatureTensor &features_from_input,
                   FeatureTensor &upstream_features) override
        {
            (void)gradient_from_output;
            (void)features_from_input;
            (void)upstream_features;

            // Target an activation distribution (neutral + bounded)
            const float targetMean = 0.0f;
            const float targetVar = 0.25f; // std 0.5

            // beta corrects mean
            adapt_.mu = 1e-4f;
            adapt_.minV = -2.0f;
            adapt_.maxV = 2.0f;
            const float meanErr = (targetMean - outMean_.mean);
            for (int i = 0; i < beta.size(); ++i)
                adapt_.step(beta[i], beta[i] + meanErr);

            // gamma corrects variance (scale)
            const float ratio = std::sqrt((targetVar + 1e-6f) / (outVar_.mean + 1e-6f));
            adapt_.mu = 5e-5f;
            adapt_.minV = 0.05f;
            adapt_.maxV = 5.0f;
            for (int i = 0; i < gamma.size(); ++i)
                adapt_.step(gamma[i], gamma[i] * ratio);

            // alpha: VERY slow drift toward 0.5 (keeps EMA stable)
            adaptAlpha_.mu = 1e-6f;
            adaptAlpha_.minV = 0.001f;
            adaptAlpha_.maxV = 0.999f;
            for (int i = 0; i < learnable_alpha.size(); ++i)
                adaptAlpha_.step(learnable_alpha[i], 0.5f);
        }

        const Eigen::MatrixXf &getWeights() const { return weights; }
        const Eigen::VectorXf &getGamma() const { return gamma; }
        const Eigen::VectorXf &getBeta() const { return beta; }
        const Eigen::VectorXf &getAlpha() const { return learnable_alpha; }

        void setWeights(const Eigen::MatrixXf &W) { weights = W; }
        void setGamma(const Eigen::VectorXf &G) { gamma = G; }
        void setBeta(const Eigen::VectorXf &B) { beta = B; }
        void setAlpha(const Eigen::VectorXf &A) { learnable_alpha = A; }

        void reset() override
        {
            input_ema_state.setZero();
            last_input_ema.setZero();
            last_preact.setZero();
            last_postnorm.setZero();
            last_activation.setZero();
            last_xhat.setZero();
            last_external_input.setZero();
            outMean_.reset(0.0f, 1.0f);
            outVar_.reset(0.0f, 1.0f);
        }

        size_t getNumParams() const override
        {
            const size_t w = static_cast<size_t>(weights.rows()) * static_cast<size_t>(weights.cols());
            const size_t g = static_cast<size_t>(gamma.size());
            const size_t b = static_cast<size_t>(beta.size());
            const size_t a = static_cast<size_t>(learnable_alpha.size());
            return w + g + b + a;
        }

    private:
        int input_size;
        int output_size;
        ActivationType activation_type;
        bool use_layer_norm;

        Eigen::MatrixXf weights; // [output_size][input_size]
        Eigen::VectorXf gamma;   // LN scale
        Eigen::VectorXf beta;    // LN shift (and bias if LN off)

        Eigen::VectorXf input_ema_state;
        Eigen::VectorXf learnable_alpha;

        Eigen::VectorXf last_input_ema;
        Eigen::VectorXf last_preact;
        Eigen::VectorXf last_postnorm;
        Eigen::VectorXf last_xhat;
        Eigen::VectorXf last_activation;
        Eigen::VectorXf last_external_input;

        // agent stats + adapt
        ChromaUtils::RunningStats outMean_;
        ChromaUtils::RunningStats outVar_;
        ChromaUtils::IIRAdapt adapt_;
        ChromaUtils::IIRAdapt adaptAlpha_;

        SGDWithMomentum optimizer;
        SGDWithMomentum optimizer_alpha;
    };

    // ============================================================================
    // attentionLayer
    // Agent-mode: keep projections stable, adapt only small output affine.
    // ============================================================================
    class attentionLayer : public DifferentiableModule
    {
    public:
        attentionLayer(int inputSize,
                       int outputSize,
                       int numHeads = 4,
                       ActivationType activationType = ActivationType::Linear,
                       bool useLayerNorm = true)
            : heads(numHeads), d_model(outputSize), d_k(outputSize / numHeads)
        {
            assert(outputSize % numHeads == 0);
            Wq = Eigen::MatrixXf(outputSize, inputSize);
            Wk = Eigen::MatrixXf(outputSize, inputSize);
            Wv = Eigen::MatrixXf(outputSize, inputSize);
            Wo = Eigen::MatrixXf(outputSize, outputSize);

            initMatrix(Wq);
            initMatrix(Wk);
            initMatrix(Wv);
            initMatrix(Wo);

            queryLayer = std::make_unique<ChromaFlow::denseLayer>(inputSize, outputSize, activationType, useLayerNorm, 0.0005f, 0.01f);
            keyLayer = std::make_unique<ChromaFlow::denseLayer>(inputSize, outputSize, activationType, useLayerNorm, 0.0005f, 0.01f);
            valueLayer = std::make_unique<ChromaFlow::denseLayer>(inputSize, outputSize, activationType, useLayerNorm, 0.0005f, 0.01f);

            out_gain = Eigen::VectorXf::Ones(outputSize);
            out_bias = Eigen::VectorXf::Zero(outputSize);
            outMean_.reset(0.0f, 1.0f);
            outVar_.reset(0.0f, 1.0f);
        }

        FeatureTensor forward(const FeatureTensor &input) override
        {
            last_input = input.data;

            // single-vector case
            if (input.data.rows() <= 1)
            {
                FeatureTensor qT = queryLayer->forward(input);
                FeatureTensor kT = keyLayer->forward(input);
                FeatureTensor vT = valueLayer->forward(input);

                Eigen::VectorXf q = qT.data.row(0).transpose();
                Eigen::VectorXf k = kT.data.row(0).transpose();
                Eigen::VectorXf v = vT.data.row(0).transpose();

                Eigen::VectorXf scores = (q.array() * k.array()) / std::sqrt((float)std::max(1, (int)q.size()));
                Eigen::VectorXf expS = (scores.array() - scores.maxCoeff()).exp().matrix();
                const float denom = expS.sum();

                Eigen::VectorXf att;
                if (denom > 0.0f)
                    att = (expS.array() / denom).matrix();
                else
                    att = Eigen::VectorXf::Constant(expS.size(), 1.0f / std::max(1, (int)expS.size()));

                Eigen::VectorXf res = att.cwiseProduct(v);

                // agent affine (small coeff set)
                res = res.cwiseProduct(out_gain) + out_bias;

                // stats
                const float m = res.mean();
                const float vv = static_cast<float>((res.array() - m).square().mean());
                outMean_.push(m);
                outVar_.push(vv);

                last_output_nonseq = res;

                FeatureTensor out;
                out.data.resize(1, static_cast<Eigen::Index>(res.size()));
                out.data.row(0) = res.transpose();
                out.numSamples = 1;
                out.features = static_cast<int>(res.size());
                return out;
            }

            // sequence case (kept, but not your primary use case)
            const int seq_len = static_cast<int>(input.data.rows());
            const int d_in = static_cast<int>(input.data.cols());
            (void)d_in;

            Q = Eigen::MatrixXf(seq_len, d_model);
            K = Eigen::MatrixXf(seq_len, d_model);
            V = Eigen::MatrixXf(seq_len, d_model);
            for (int t = 0; t < seq_len; ++t)
            {
                Eigen::Map<const Eigen::VectorXf> xRow(input.data.row(t).data(), input.data.cols());
                Q.row(t) = (Wq * xRow).transpose();
                K.row(t) = (Wk * xRow).transpose();
                V.row(t) = (Wv * xRow).transpose();
            }

            output_mat = Eigen::MatrixXf::Zero(seq_len, d_model);
            attn_weights = std::vector<Eigen::MatrixXf>(heads);
            const float scale = 1.0f / std::sqrt((float)d_k);

            for (int h = 0; h < heads; ++h)
            {
                const int offset = h * d_k;
                Eigen::MatrixXf Qh = Q.block(0, offset, seq_len, d_k);
                Eigen::MatrixXf Kh = K.block(0, offset, seq_len, d_k);
                Eigen::MatrixXf Vh = V.block(0, offset, seq_len, d_k);

                Eigen::MatrixXf scores = (Qh * Kh.transpose()) * scale;

                Eigen::MatrixXf softmaxed(seq_len, seq_len);
                for (int i = 0; i < seq_len; ++i)
                {
                    Eigen::VectorXf row = scores.row(i).transpose();
                    float m = row.maxCoeff();
                    row = (row.array() - m).exp();
                    float s = row.sum();
                    if (s > 0.0f)
                        row /= s;
                    softmaxed.row(i) = row.transpose();
                    Eigen::RowVectorXf weighted = row.transpose() * Vh;
                    output_mat.block(i, offset, 1, d_k) = weighted;
                }
                attn_weights[h] = softmaxed;
            }

            final_out = Eigen::MatrixXf(seq_len, d_model);
            for (int t = 0; t < seq_len; ++t)
            {
                Eigen::Map<const Eigen::VectorXf> outRow(output_mat.row(t).data(), output_mat.cols());
                final_out.row(t) = (Wo * outRow).transpose();
            }

            // apply agent affine per timestep
            for (int t = 0; t < seq_len; ++t)
                final_out.row(t) = (final_out.row(t).transpose().cwiseProduct(out_gain) + out_bias).transpose();

            last_output_vec = final_out;

            FeatureTensor outFt;
            outFt.data = final_out;
            outFt.numSamples = seq_len;
            outFt.features = d_model;
            return outFt;
        }

        // Backprop retained (debug/tests). In agent mode we do not update big matrices.
        std::pair<FeatureTensor, FeatureTensor> backward(const FeatureTensor &grad_output)
        {
            FeatureTensor grad_prev;
            FeatureTensor act_prev;

            if (grad_output.data.size() == 0)
            {
                grad_prev.data = Eigen::MatrixXf::Zero(last_input.rows(), last_input.cols());
                grad_prev.numSamples = last_input.rows();
                grad_prev.features = last_input.cols();

                act_prev.data = last_input;
                act_prev.numSamples = last_input.rows();
                act_prev.features = last_input.cols();
                return {grad_prev, act_prev};
            }

            const int seq_len = static_cast<int>(last_input.rows());
            const int inDim = static_cast<int>(last_input.cols());

            if (seq_len <= 1)
            {
                Eigen::VectorXf g;
                if (grad_output.data.rows() == 1)
                    g = grad_output.data.row(0).transpose();
                else
                    g = grad_output.data.col(0);

                Eigen::MatrixXf Wproj = Eigen::MatrixXf::Zero(d_model, inDim);
                int count = 0;
                if (queryLayer)
                {
                    Wproj += queryLayer->getWeights();
                    ++count;
                }
                if (keyLayer)
                {
                    Wproj += keyLayer->getWeights();
                    ++count;
                }
                if (valueLayer)
                {
                    Wproj += valueLayer->getWeights();
                    ++count;
                }
                if (count > 0)
                    Wproj /= (float)count;

                Eigen::VectorXf gradInputVec = Wproj.transpose() * g;

                grad_prev.data = Eigen::MatrixXf::Zero(inDim, 1);
                grad_prev.data.col(0) = gradInputVec;
                grad_prev.numSamples = inDim;
                grad_prev.features = 1;

                act_prev.data = last_input;
                act_prev.numSamples = last_input.rows();
                act_prev.features = last_input.cols();

#if !CHROMAFLOW_AGENT_MODE
                if (training_allowed && last_output_nonseq.size() == g.size())
                {
                    Eigen::MatrixXf gradWo = g * last_output_nonseq.transpose();
                    Wo -= 1e-6f * gradWo;
                }
#endif
                return {grad_prev, act_prev};
            }

            const int d_model_local = d_model;
            jassert(grad_output.data.rows() == seq_len);
            jassert(grad_output.data.cols() == d_model_local);

            Eigen::MatrixXf dOutMat = grad_output.data * Wo;

            Eigen::MatrixXf Wavg = (Wq.transpose() + Wk.transpose() + Wv.transpose()) / 3.0f;
            Eigen::MatrixXf gradInput = dOutMat * Wavg.transpose();

            grad_prev.data = gradInput;
            grad_prev.numSamples = seq_len;
            grad_prev.features = inDim;

            act_prev.data = last_input;
            act_prev.numSamples = seq_len;
            act_prev.features = inDim;

#if !CHROMAFLOW_AGENT_MODE
            Eigen::MatrixXf gradWo = Eigen::MatrixXf::Zero(Wo.rows(), Wo.cols());
            for (int t = 0; t < seq_len; ++t)
            {
                Eigen::VectorXf gout = grad_output.data.row(t).transpose();
                Eigen::VectorXf outv = output_mat.row(t).transpose();
                gradWo += gout * outv.transpose();
            }
            if (training_allowed)
                Wo -= 1e-6f * gradWo;
#endif

            return {grad_prev, act_prev};
        }

        // Agent learning: adapt only out_gain/out_bias (small coefficient set)
        void learn(const FeatureTensor &gradient_from_output,
                   const FeatureTensor &features_from_input,
                   FeatureTensor &upstream_features) override
        {
            (void)gradient_from_output;
            (void)features_from_input;
            (void)upstream_features;

            const float targetMean = 0.0f;
            const float targetVar = 0.25f;

            // bias corrects mean
            adapt_.mu = 1e-4f;
            adapt_.minV = -2.0f;
            adapt_.maxV = 2.0f;
            const float meanErr = (targetMean - outMean_.mean);
            for (int i = 0; i < out_bias.size(); ++i)
                adapt_.step(out_bias[i], out_bias[i] + meanErr);

            // gain corrects variance
            const float ratio = std::sqrt((targetVar + 1e-6f) / (outVar_.mean + 1e-6f));
            adapt_.mu = 5e-5f;
            adapt_.minV = 0.05f;
            adapt_.maxV = 5.0f;
            for (int i = 0; i < out_gain.size(); ++i)
                adapt_.step(out_gain[i], out_gain[i] * ratio);
        }

        void reset() override
        {
            last_input.setZero();
            Q.setZero();
            K.setZero();
            V.setZero();
            output_mat.setZero();
            final_out.setZero();
            last_output_nonseq.setZero();
            out_gain.setOnes();
            out_bias.setZero();
            outMean_.reset(0.0f, 1.0f);
            outVar_.reset(0.0f, 1.0f);
        }

        size_t getNumParams() const override
        {
            size_t total = static_cast<size_t>(Wq.size()) + static_cast<size_t>(Wk.size()) +
                           static_cast<size_t>(Wv.size()) + static_cast<size_t>(Wo.size());
            if (queryLayer)
                total += queryLayer->getNumParams();
            if (keyLayer)
                total += keyLayer->getNumParams();
            if (valueLayer)
                total += valueLayer->getNumParams();
            total += static_cast<size_t>(out_gain.size()) + static_cast<size_t>(out_bias.size());
            return total;
        }

        const ChromaFlow::denseLayer *getQueryLayer() const { return queryLayer.get(); }
        const ChromaFlow::denseLayer *getKeyLayer() const { return keyLayer.get(); }
        const ChromaFlow::denseLayer *getValueLayer() const { return valueLayer.get(); }
        void setTrainingAllowed(bool allowed) { training_allowed = allowed; }

    private:
        int heads;
        int d_model;
        int d_k;

        Eigen::MatrixXf Wq, Wk, Wv, Wo;
        Eigen::MatrixXf last_input;
        Eigen::MatrixXf Q, K, V;
        Eigen::MatrixXf output_mat;
        Eigen::MatrixXf final_out;

        Eigen::VectorXf last_output_nonseq;
        Eigen::MatrixXf last_output_vec;
        std::vector<Eigen::MatrixXf> attn_weights;

        std::unique_ptr<ChromaFlow::denseLayer> queryLayer;
        std::unique_ptr<ChromaFlow::denseLayer> keyLayer;
        std::unique_ptr<ChromaFlow::denseLayer> valueLayer;

        bool training_allowed = false;

        // agent small coeff set
        Eigen::VectorXf out_gain;
        Eigen::VectorXf out_bias;
        ChromaUtils::RunningStats outMean_;
        ChromaUtils::RunningStats outVar_;
        ChromaUtils::IIRAdapt adapt_;

        static void initMatrix(Eigen::MatrixXf &M)
        {
            const float limit = std::sqrt(6.0f / (M.rows() + M.cols()));
            for (Eigen::Index r = 0; r < M.rows(); ++r)
                for (Eigen::Index c = 0; c < M.cols(); ++c)
                    M(r, c) = ChromaUtils::randUniform(-limit, limit);
        }
    };

    // ============================================================================
    // RNNCell
    // Agent-mode: explicit stable IIR-ish state update with trainable leak.
    // ============================================================================
    class RNNCell : public DifferentiableModule
    {
    public:
        RNNCell(int inputSize, int hiddenSize, float learningRate = 0.005f, float momentum = 0.9f)
            : input_size(inputSize),
              hidden_size(hiddenSize),
              W_x(hiddenSize, inputSize),
              W_h(hiddenSize, hiddenSize),
              b(Eigen::VectorXf::Zero(hiddenSize)),
              hidden_state(Eigen::VectorXf::Zero(hiddenSize)),
              last_input(Eigen::VectorXf::Zero(inputSize)),
              last_hidden_prev(Eigen::VectorXf::Zero(hiddenSize)),
              optimizer(learningRate, momentum)
        {
            const float scale_x = std::sqrt(1.0f / static_cast<float>(input_size));
            const float scale_h = std::sqrt(1.0f / static_cast<float>(hidden_size));
            initMatrix(W_x, scale_x);
            initMatrix(W_h, scale_h);
            b.setZero();

            leak_ = 0.05f; // IIR-ish state smoothing
            outRms_.reset(0.0f, 1.0f);
        }

        FeatureTensor forward(const FeatureTensor &x_t) override
        {
            FeatureTensor out;
            const int cols = x_t.data.rows() > 0 ? static_cast<int>(x_t.data.cols()) : 0;
            if (cols < input_size)
            {
                out.numSamples = 1;
                out.features = hidden_size;
                out.data.resize(1, static_cast<Eigen::Index>(hidden_size));
                out.data.setZero();
                return out;
            }

            last_hidden_prev = hidden_state;
            last_input = x_t.data.row(0).transpose();

            Eigen::VectorXf z = W_x * last_input + W_h * last_hidden_prev + b;
            last_preact = z;

            // candidate
            Eigen::VectorXf cand = z.array().tanh().matrix();

            // stable IIR-ish update
            hidden_state = (1.0f - leak_) * last_hidden_prev + leak_ * cand;

            last_output_vec = hidden_state;

            // stats
            const float rms = std::sqrt(hidden_state.squaredNorm() / std::max(1, (int)hidden_state.size()));
            outRms_.push(rms);

            out.data.resize(1, static_cast<Eigen::Index>(hidden_state.size()));
            out.data.row(0) = hidden_state.transpose();
            out.numSamples = 1;
            out.features = hidden_size;
            return out;
        }

        // Backprop retained (debug/tests). Agent mode does not need it for learning.
        std::pair<FeatureTensor, FeatureTensor> backward(const FeatureTensor &grad_output)
        {
            FeatureTensor grad_prev, act_prev;

            if (grad_output.data.size() == 0)
            {
                grad_prev.data = Eigen::MatrixXf::Zero(last_input.size(), 1);
                grad_prev.numSamples = static_cast<int>(last_input.size());
                grad_prev.features = 1;

                act_prev.data = last_input;
                act_prev.numSamples = static_cast<int>(last_input.size());
                act_prev.features = 1;
                return {grad_prev, act_prev};
            }

            const int goRows = grad_output.data.rows();
            const int goCols = grad_output.data.cols();

            Eigen::VectorXf grad;
            if (goRows == 1 && goCols == hidden_size)
                grad = grad_output.data.row(0).transpose();
            else if (goCols == 1 && goRows == hidden_size)
                grad = grad_output.data.col(0);
            else if (goRows == 1 || goCols == 1)
            {
                const int len = std::max(goRows, goCols);
                jassert(len == hidden_size);
                grad = (goRows == 1) ? grad_output.data.row(0).transpose().eval() : grad_output.data.col(0).eval();
            }
            else
            {
                jassertfalse;
                grad = Eigen::VectorXf::Zero(hidden_size);
            }

            jassert(grad.size() == hidden_size);
            jassert(last_output_vec.size() == hidden_size);

            Eigen::VectorXf dtanh = (1.0f - last_output_vec.array().square()).matrix();
            Eigen::VectorXf dPre = grad.cwiseProduct(dtanh);

            Eigen::MatrixXf grad_Wx = dPre * last_input.transpose();
            Eigen::MatrixXf grad_Wh = dPre * last_hidden_prev.transpose();
            Eigen::VectorXf grad_b = dPre;

            Eigen::VectorXf dInput = W_x.transpose() * dPre;

#if !CHROMAFLOW_AGENT_MODE
            optimizer.update(W_x, -grad_Wx, grad_output);
            optimizer.update(W_h, -grad_Wh, grad_output);
            optimizer.update(b, -grad_b, grad_output);
#endif

            grad_prev.data = Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(dInput.size()), 1);
            grad_prev.data.col(0) = dInput;
            grad_prev.numSamples = static_cast<int>(dInput.size());
            grad_prev.features = 1;

            act_prev.data = last_input;
            act_prev.numSamples = static_cast<int>(last_input.size());
            act_prev.features = 1;

            return {grad_prev, act_prev};
        }

        // Agent learning: adjust leak slowly to keep RMS in band (explicit feedback)
        void learn(const FeatureTensor &gradient_from_output,
                   const FeatureTensor &features_from_input,
                   FeatureTensor &upstream_features) override
        {
            (void)gradient_from_output;
            (void)features_from_input;
            (void)upstream_features;

            const float targetRms = 0.10f;

            adaptLeak_.mu = 1e-5f;
            adaptLeak_.minV = 0.001f;
            adaptLeak_.maxV = 0.2f;

            // If RMS is too high, reduce leak (slower state reaction). If too low, increase leak.
            adaptLeak_.step(leak_, leak_ + (targetRms - outRms_.mean));
        }

        void reset() override
        {
            hidden_state.setZero();
            last_input.setZero();
            last_hidden_prev.setZero();
            leak_ = 0.05f;
            outRms_.reset(0.0f, 1.0f);
        }

        const Eigen::MatrixXf &getWx() const { return W_x; }
        const Eigen::MatrixXf &getWh() const { return W_h; }
        const Eigen::VectorXf &getB() const { return b; }
        const Eigen::VectorXf &getHiddenState() const { return hidden_state; }

        void setWx(const Eigen::MatrixXf &M) { W_x = M; }
        void setWh(const Eigen::MatrixXf &M) { W_h = M; }
        void setB(const Eigen::VectorXf &V) { b = V; }
        void setHiddenState(const Eigen::VectorXf &H) { hidden_state = H; }

        size_t getNumParams() const override
        {
            const size_t wx = static_cast<size_t>(W_x.rows()) * static_cast<size_t>(W_x.cols());
            const size_t wh = static_cast<size_t>(W_h.rows()) * static_cast<size_t>(W_h.cols());
            const size_t vb = static_cast<size_t>(b.size());
            return wx + wh + vb + 1u; // + leak
        }

    private:
        int input_size;
        int hidden_size;

        Eigen::MatrixXf W_x;
        Eigen::MatrixXf W_h;
        Eigen::VectorXf b;

        Eigen::VectorXf hidden_state;
        Eigen::VectorXf last_input;
        Eigen::VectorXf last_hidden_prev;

        Eigen::VectorXf last_preact;
        Eigen::VectorXf last_output_vec;

        // agent
        float leak_ = 0.05f;
        ChromaUtils::RunningStats outRms_;
        ChromaUtils::IIRAdapt adaptLeak_;

        SGDWithMomentum optimizer;

        static void initMatrix(Eigen::MatrixXf &M, float scale)
        {
            uint32_t seed = 1234567u;
            auto nextU = [&seed]()
            {
                seed = 1664525u * seed + 1013904223u;
                return static_cast<float>(seed) / static_cast<float>(UINT32_MAX);
            };
            for (Eigen::Index r = 0; r < M.rows(); ++r)
                for (Eigen::Index c = 0; c < M.cols(); ++c)
                {
                    const float u = nextU();
                    M(r, c) = (2.0f * u - 1.0f) * scale;
                }
        }
    };

} // namespace ChromaFlow
