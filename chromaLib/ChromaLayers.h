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
    // ==========================================================

    // Agent mode helpers (forward-only, stable IIR-ish updates)
    // ==========================================================

    static inline float clampf(float v, float lo, float hi)
    {
        return std::max(lo, std::min(hi, v));
    }

    static inline float finiteOr(float v, float fallback = 0.0f)
    {
        return std::isfinite(v) ? v : fallback;
    }

    static inline float meanAll(const FeatureTensor &t)
    {
        if (t.data.size() == 0)
            return 0.0f;
        float s = 0.0f;
        int n = 0;
        for (Eigen::Index r = 0; r < t.data.rows(); ++r)
            for (Eigen::Index c = 0; c < t.data.cols(); ++c)
            {
                s += finiteOr(t.data(r, c), 0.0f);
                ++n;
            }
        return n > 0 ? (s / (float)n) : 0.0f;
    }

    // Scalar affine agent (good for conv + global blocks)
    struct AgentAffineScalar
    {
        // affine
        float gain = 1.0f;
        float bias = 0.0f;

        // IIR error state
        float eLP = 0.0f;

        // rates (tune)
        float aErr = 0.01f;   // error IIR pole
        float muGain = 1e-4f; // gain adaptation
        float muBias = 1e-4f; // bias adaptation
        float leak = 1e-5f;   // drift to neutral

        // clamps
        float gMin = 0.25f, gMax = 4.0f;
        float bMin = -2.0f, bMax = 2.0f;

        inline void apply(float &v) const noexcept
        {
            v = v * gain + bias;
        }

        inline void updateFromError(float e) noexcept
        {
            e = finiteOr(e, 0.0f);
            eLP = (1.0f - aErr) * eLP + aErr * e;

            // leak-to-neutral
            gain = (1.0f - leak) * gain + leak * 1.0f;
            bias = (1.0f - leak) * bias;

            // stable gain update (log domain)
            gain *= std::exp(muGain * eLP);
            bias += muBias * eLP;

            gain = clampf(gain, gMin, gMax);
            bias = clampf(bias, bMin, bMax);
        }
    };

    // Vector bias agent (dense/RNN outputs) with shared gain
    struct AgentAffineVector
    {
        float gain = 1.0f;
        Eigen::VectorXf bias; // per-dimension bias
        Eigen::VectorXf eLP;  // per-dimension IIR error

        float aErr = 0.02f;
        float muGain = 1e-4f;
        float muBias = 5e-5f;
        float leak = 1e-5f;

        float gMin = 0.25f, gMax = 4.0f;
        float bMin = -2.0f, bMax = 2.0f;

        void resizeIfNeeded(int dim)
        {
            if ((int)bias.size() != dim)
            {
                bias = Eigen::VectorXf::Zero(dim);
                eLP = Eigen::VectorXf::Zero(dim);
            }
        }

        inline void apply(Eigen::VectorXf &v) const noexcept
        {
            // v := v*gain + bias
            if (bias.size() == v.size())
                v = (v.array() * gain + bias.array()).matrix();
            else
                v = (v.array() * gain).matrix();
        }

        inline void updateFromErrorVec(const Eigen::VectorXf &e)
        {
            const int dim = (int)e.size();
            if (dim <= 0)
                return;
            resizeIfNeeded(dim);

            // IIR error
            for (int i = 0; i < dim; ++i)
            {
                const float ei = finiteOr(e(i), 0.0f);
                eLP(i) = (1.0f - aErr) * eLP(i) + aErr * ei;
            }

            // leak-to-neutral
            gain = (1.0f - leak) * gain + leak * 1.0f;
            bias = (1.0f - leak) * bias;

            // shared gain update from mean error
            const float m = eLP.size() ? (float)eLP.mean() : 0.0f;
            gain *= std::exp(muGain * m);

            // per-dim bias update
            bias += (muBias * eLP.array()).matrix();

            gain = clampf(gain, gMin, gMax);
            for (int i = 0; i < bias.size(); ++i)
                bias(i) = clampf(bias(i), bMin, bMax);
        }
    };

    // build error vector from FeatureTensor (row0 preferred)
    static inline Eigen::VectorXf errorVectorRow0(const FeatureTensor &err)
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

    enum class ActivationType
    {
        LeakyRelu,
        Tanh,
        Sigmoid,
        Linear
    };

    class convolutionalLayer : public DifferentiableModule
    {
    public:
        convolutionalLayer(int /*inputChannels*/, int /*outputChannels*/, int kernelSize)
        {
            kernelSize_ = std::max(1, kernelSize);
            kernel_ = Eigen::VectorXf::Zero(kernelSize_);
            bias_ = 0.0f;

            for (int i = 0; i < kernelSize_; ++i)
                kernel_(i) = ChromaUtils::randUniform(-0.01f, 0.01f);
        }

        FeatureTensor forward(const FeatureTensor &input) override
        {
            last_input = input.data;
            return forwardconvolve(input);
        }

        FeatureTensor forwardconvolve(const FeatureTensor &input)
        {
            const int K = kernelSize_;
            if (K <= 0 || input.data.size() == 0)
                return input;

            // expects Nx1
            const int inN = (int)input.data.rows();
            if (input.data.cols() < 1 || inN <= 0)
                return input;

            const int pad = K / 2;
            const int validOut = std::max(0, inN - K + 1);
            const int outN = validOut + 2 * pad;

            FeatureTensor output;
            output.data = Eigen::MatrixXf::Zero(outN, 1);
            output.numSamples = outN;
            output.features = 1;

            Eigen::VectorXf kr = kernel_.reverse();
            const float *dataPtr = input.data.col(0).data();
            float *outPtr = output.data.data();

            // valid conv written into [pad .. pad+validOut)
            for (int i = 0; i < validOut; ++i)
            {
                Eigen::Map<const Eigen::VectorXf> seg(dataPtr + i, K);
                outPtr[i + pad] = seg.dot(kr);
            }

            // bias + agent affine (no clamp here; clamp at param mapping stage)
            for (int i = 0; i < outN; ++i)
            {
                float v = outPtr[i] + bias_;
                agent_.apply(v);
                outPtr[i] = v;
            }

            last_output = output.data;
            return output;
        }

        // AGENT MODE: treat "gradient_from_output" as error (target - output)
        void learn(const FeatureTensor &gradient_from_output,
                   const FeatureTensor & /*features_from_input*/,
                   FeatureTensor & /*upstream_features*/) override
        {
            agent_.updateFromError(meanAll(gradient_from_output));
        }

        void reset() override
        {
            last_input.setZero();
            last_output.setZero();
            agent_ = AgentAffineScalar{};
        }

        // serialization
        Eigen::MatrixXf getWeights() const
        {
            Eigen::MatrixXf W(kernel_.size(), 1);
            W.col(0) = kernel_;
            return W;
        }
        Eigen::VectorXf getBiases() const
        {
            Eigen::VectorXf b(1);
            b(0) = bias_;
            return b;
        }
        void setWeights(const Eigen::MatrixXf &W)
        {
            if (W.size() <= 0)
                return;
            kernelSize_ = (int)W.rows();
            kernel_ = W.col(0);
        }
        void setBiases(const Eigen::VectorXf &b)
        {
            bias_ = (b.size() > 0) ? b(0) : 0.0f;
        }

        // convenience setter used by tests
        void setKernel(const Eigen::VectorXf &k)
        {
            if (k.size() <= 0)
                return;
            kernel_ = k;
            kernelSize_ = (int)k.size();
        }

        size_t getNumParams() const override
        {
            // real params + agent (gain+bias)
            return (size_t)kernel_.size() + 1 + 2;
        }

    private:
        Eigen::VectorXf kernel_;
        float bias_ = 0.0f;
        int kernelSize_ = 1;

        Eigen::MatrixXf last_input;
        Eigen::MatrixXf last_output;

        AgentAffineScalar agent_;
    };

    class denseLayer : public DifferentiableModule
    {
    public:
        denseLayer(int inputSize,
                   int outputSize,
                   ActivationType activation = ActivationType::LeakyRelu,
                   bool useLayerNorm = true)
            : input_size(inputSize),
              output_size(outputSize),
              activation_type(activation),
              use_layer_norm(useLayerNorm)
        {
            // Xavier-ish init
            const float limit = std::sqrt(6.0f / (input_size + output_size));
            weights.resize(output_size, input_size);
            for (int o = 0; o < output_size; ++o)
                for (int i = 0; i < input_size; ++i)
                    weights(o, i) = ChromaUtils::randUniform(-limit, limit);

            // LN params (static / non-learning here)
            gamma = Eigen::VectorXf::Ones(output_size);
            beta = Eigen::VectorXf::Zero(output_size);

            // input EMA for stability (fixed alpha)
            input_ema_state = Eigen::VectorXf::Zero(input_size);
            alpha_ema = 0.2f;

            agent_.resizeIfNeeded(output_size);
        }

        FeatureTensor forward(const FeatureTensor &upstream_features) override
        {
            // Read as 1×N or N×1 → input_size vector
            Eigen::VectorXf x = Eigen::VectorXf::Zero(input_size);
            if (upstream_features.data.size() != 0)
            {
                if (upstream_features.data.rows() == 1)
                {
                    const int n = std::min((int)upstream_features.data.cols(), input_size);
                    x.head(n) = upstream_features.data.row(0).transpose().head(n);
                }
                else if (upstream_features.data.cols() == 1)
                {
                    const int n = std::min((int)upstream_features.data.rows(), input_size);
                    x.head(n) = upstream_features.data.col(0).head(n);
                }
            }

            // EMA stabilize
            input_ema_state = (1.0f - alpha_ema) * input_ema_state + alpha_ema * x;
            last_input_ema = input_ema_state;

            Eigen::VectorXf z = (weights * input_ema_state).eval();
            last_preact = z;

            if (use_layer_norm)
            {
                const float mean = z.mean();
                const float var = std::max(1e-6f,
                                           (float)((z.array() - mean).square().mean()));
                Eigen::VectorXf norm = ((z.array() - mean) / std::sqrt(var)).matrix();
                z = (gamma.cwiseProduct(norm) + beta).eval();
            }

            last_postnorm = z;

            Eigen::VectorXf a = z.unaryExpr([this](float v)
                                            {
            switch (activation_type)
            {
                case ActivationType::LeakyRelu: return v >= 0.0f ? v : 0.01f * v;
                case ActivationType::Tanh:      return std::tanh(v);
                case ActivationType::Sigmoid:   return 1.0f / (1.0f + std::exp(-v));
                case ActivationType::Linear:    return v;
                default:                        assert(false); return v; // should not happen
            } })
                                    .eval();

            // agent affine (shared gain + per-dim bias)
            agent_.resizeIfNeeded((int)a.size());
            agent_.apply(a);

            last_activation = a;

            FeatureTensor out;
            out.data.resize(1, (Eigen::Index)a.size());
            out.data.row(0) = a.transpose();
            out.numSamples = 1;
            out.features = (int)a.size();
            return out;
        }

        // AGENT MODE: "gradient_from_output" == error (target - output)
        void learn(const FeatureTensor &gradient_from_output,
                   const FeatureTensor & /*features_from_input*/,
                   FeatureTensor & /*upstream_features*/) override
        {
            Eigen::VectorXf e = errorVectorRow0(gradient_from_output);
            if (e.size() <= 0)
                return;

            // if error dims don’t match, truncate/expand safely
            if (e.size() != output_size)
            {
                Eigen::VectorXf e2 = Eigen::VectorXf::Zero(output_size);
                const int n = std::min((int)e.size(), output_size);
                e2.head(n) = e.head(n);
                e = e2;
            }

            agent_.updateFromErrorVec(e);
        }

        void reset() override
        {
            input_ema_state.setZero();
            last_input_ema.setZero();
            last_preact.setZero();
            last_postnorm.setZero();
            last_activation.setZero();
            agent_ = AgentAffineVector{};
            agent_.resizeIfNeeded(output_size);
        }

        // serialization
        const Eigen::MatrixXf &getWeights() const { return weights; }
        const Eigen::VectorXf &getGamma() const { return gamma; }
        const Eigen::VectorXf &getBeta() const { return beta; }

        void setWeights(const Eigen::MatrixXf &W) { weights = W; }
        void setGamma(const Eigen::VectorXf &G) { gamma = G; }
        void setBeta(const Eigen::VectorXf &B) { beta = B; }

        size_t getNumParams() const override
        {
             const size_t w = (size_t)weights.size();
            const size_t g = (size_t)gamma.size();
            const size_t b = (size_t)beta.size();
            // agent: gain + biasVec
            const size_t a = 1 + (size_t)output_size;
            return w + g + b + a;
        }

    private:
        int input_size = 0;
        int output_size = 0;
        ActivationType activation_type = ActivationType::LeakyRelu;
        bool use_layer_norm = true;

        Eigen::MatrixXf weights;
        Eigen::VectorXf gamma, beta;

        float alpha_ema = 0.2f;
        Eigen::VectorXf input_ema_state;

        Eigen::VectorXf last_input_ema;
        Eigen::VectorXf last_preact;
        Eigen::VectorXf last_postnorm;
        Eigen::VectorXf last_activation;

        AgentAffineVector agent_;
    };

    class attentionLayer : public DifferentiableModule
    {
    public:
        attentionLayer(int inputSize,
                       int outputSize,
                       int numHeads = 4,
                       ActivationType activationType = ActivationType::Linear,
                       bool useLayerNorm = true)
            : heads(numHeads),
              d_model(outputSize),
              d_k(std::max(1, outputSize / std::max(1, numHeads)))
        {
            jassert(outputSize % numHeads == 0);

            queryLayer = std::make_unique<ChromaFlow::denseLayer>(inputSize, outputSize, activationType, useLayerNorm);
            keyLayer = std::make_unique<ChromaFlow::denseLayer>(inputSize, outputSize, activationType, useLayerNorm);
            valueLayer = std::make_unique<ChromaFlow::denseLayer>(inputSize, outputSize, activationType, useLayerNorm);

            agent_.resizeIfNeeded(outputSize);
        }

        FeatureTensor forward(const FeatureTensor &input) override
        {
            last_input = input.data;

            // SINGLE VECTOR path (your use case)
            if (input.data.rows() <= 1)
            {
                FeatureTensor qT = queryLayer->forward(input);
                FeatureTensor kT = keyLayer->forward(input);
                FeatureTensor vT = valueLayer->forward(input);

                Eigen::VectorXf q = qT.data.row(0).transpose();
                Eigen::VectorXf k = kT.data.row(0).transpose();
                Eigen::VectorXf v = vT.data.row(0).transpose();

                // elementwise “dot attention”
                Eigen::VectorXf scores = (q.array() * k.array()) / std::sqrt((float)std::max(1, (int)q.size()));
                Eigen::VectorXf expS = (scores.array() - scores.maxCoeff()).exp().matrix();
                const float denom = expS.sum();

                Eigen::VectorXf att;
                if (denom > 0.0f)
                {
                    att = (expS.array() / denom).matrix();
                }
                else
                {
                    att = Eigen::VectorXf::Constant((int)expS.size(), 1.0f / std::max(1, (int)expS.size()));
                }

                Eigen::VectorXf res = att.cwiseProduct(v);

                // agent affine
                agent_.resizeIfNeeded((int)res.size());
                agent_.apply(res);

                FeatureTensor out;
                out.data.resize(1, (Eigen::Index)res.size());
                out.data.row(0) = res.transpose();
                out.numSamples = 1;
                out.features = (int)res.size();

                last_out_vec = res;
                return out;
            }

            // SEQ path (kept, but still no backward)
            const int seq_len = (int)input.data.rows();
            FeatureTensor out = input; // fallback
            out.data = Eigen::MatrixXf::Zero(seq_len, d_model);
            out.numSamples = seq_len;
            out.features = d_model;

            // Very conservative: just apply per-row dense Q/K/V + per-row elementwise attention
            for (int t = 0; t < seq_len; ++t)
            {
                FeatureTensor step;
                step.data.resize(1, input.data.cols());
                step.data.row(0) = input.data.row(t);
                step.numSamples = 1;
                step.features = (int)input.data.cols();

                auto qT = queryLayer->forward(step);
                auto kT = keyLayer->forward(step);
                auto vT = valueLayer->forward(step);

                Eigen::VectorXf q = qT.data.row(0).transpose();
                Eigen::VectorXf k = kT.data.row(0).transpose();
                Eigen::VectorXf v = vT.data.row(0).transpose();

                Eigen::VectorXf scores = (q.array() * k.array()) / std::sqrt((float)std::max(1, (int)q.size()));
                Eigen::VectorXf expS = (scores.array() - scores.maxCoeff()).exp().matrix();
                const float denom = expS.sum();
                Eigen::VectorXf att;
                if (denom > 0.0f)
                {
                    att = (expS.array() / denom).matrix();
                }
                else
                {
                    att = Eigen::VectorXf::Constant((int)expS.size(), 1.0f / std::max(1, (int)expS.size()));
                }
                Eigen::VectorXf res = att.cwiseProduct(v);

                agent_.resizeIfNeeded((int)res.size());
                agent_.apply(res);

                out.data.row(t) = res.transpose();
            }

            return out;
        }

        // AGENT MODE: error drives only the output affine (no real attention weight training)
        void learn(const FeatureTensor &gradient_from_output,
                   const FeatureTensor & /*features_from_input*/,
                   FeatureTensor & /*upstream_features*/) override
        {
            Eigen::VectorXf e = errorVectorRow0(gradient_from_output);
            if (e.size() <= 0)
                return;

            if (e.size() != d_model)
            {
                Eigen::VectorXf e2 = Eigen::VectorXf::Zero(d_model);
                const int n = std::min((int)e.size(), d_model);
                e2.head(n) = e.head(n);
                e = e2;
            }

            agent_.updateFromErrorVec(e);
        }

        void reset() override
        {
            last_input.setZero();
            last_out_vec.setZero();
            agent_ = AgentAffineVector{};
            agent_.resizeIfNeeded(d_model);
        }

        size_t getNumParams() const override
        {
            size_t total = 0;
            if (queryLayer)
                total += queryLayer->getNumParams();
            if (keyLayer)
                total += keyLayer->getNumParams();
            if (valueLayer)
                total += valueLayer->getNumParams();
            // agent: gain + biasVec
            total += 1 + (size_t)d_model;
            return total;
        }

        const ChromaFlow::denseLayer *getQueryLayer() const { return queryLayer.get(); }
        const ChromaFlow::denseLayer *getKeyLayer() const { return keyLayer.get(); }
        const ChromaFlow::denseLayer *getValueLayer() const { return valueLayer.get(); }

    private:
        int heads = 4;
        int d_model = 0;
        int d_k = 0;

        Eigen::MatrixXf last_input;
        Eigen::VectorXf last_out_vec;

        std::unique_ptr<ChromaFlow::denseLayer> queryLayer;
        std::unique_ptr<ChromaFlow::denseLayer> keyLayer;
        std::unique_ptr<ChromaFlow::denseLayer> valueLayer;

        AgentAffineVector agent_;
    };

    class RNNCell : public DifferentiableModule
    {
    public:
        RNNCell(int inputSize, int hiddenSize)
            : input_size(inputSize),
              hidden_size(hiddenSize)
        {
            // init weights (fixed, no gradient training)
            W_x = Eigen::MatrixXf(hidden_size, input_size);
            W_h = Eigen::MatrixXf(hidden_size, hidden_size);
            b = Eigen::VectorXf::Zero(hidden_size);

            const float scale_x = std::sqrt(1.0f / std::max(1, input_size));
            const float scale_h = std::sqrt(1.0f / std::max(1, hidden_size));
            initMatrix(W_x, scale_x);
            initMatrix(W_h, scale_h);

            hidden_state = Eigen::VectorXf::Zero(hidden_size);
            last_input = Eigen::VectorXf::Zero(input_size);
            last_hidden_prev = Eigen::VectorXf::Zero(hidden_size);

            agent_.resizeIfNeeded(hidden_size);
        }

        FeatureTensor forward(const FeatureTensor &x_t) override
        {
            // expects 1×I or I×1
            Eigen::VectorXf x = Eigen::VectorXf::Zero(input_size);

            if (x_t.data.size() != 0)
            {
                if (x_t.data.rows() == 1)
                {
                    const int n = std::min((int)x_t.data.cols(), input_size);
                    x.head(n) = x_t.data.row(0).transpose().head(n);
                }
                else if (x_t.data.cols() == 1)
                {
                    const int n = std::min((int)x_t.data.rows(), input_size);
                    x.head(n) = x_t.data.col(0).head(n);
                }
            }

            last_hidden_prev = hidden_state;
            last_input = x;

            Eigen::VectorXf pre = W_x * x + W_h * last_hidden_prev + b;
            Eigen::VectorXf h = pre.array().tanh().matrix();

            // agent affine on hidden state
            agent_.resizeIfNeeded((int)h.size());
            agent_.apply(h);

            hidden_state = h;

            FeatureTensor out;
            out.data.resize(1, (Eigen::Index)hidden_state.size());
            out.data.row(0) = hidden_state.transpose();
            out.numSamples = 1;
            out.features = hidden_size;
            return out;
        }

        // AGENT MODE: error drives only the output affine (no weight training)
        void learn(const FeatureTensor &gradient_from_output,
                   const FeatureTensor & /*features_from_input*/,
                   FeatureTensor & /*upstream_features*/) override
        {
            Eigen::VectorXf e = errorVectorRow0(gradient_from_output);
            if (e.size() <= 0)
                return;

            // map error dims to hidden dims safely
            if (e.size() != hidden_size)
            {
                Eigen::VectorXf e2 = Eigen::VectorXf::Zero(hidden_size);
                const int n = std::min((int)e.size(), hidden_size);
                e2.head(n) = e.head(n);
                e = e2;
            }

            agent_.updateFromErrorVec(e);
        }

        void reset() override
        {
            hidden_state.setZero();
            last_input.setZero();
            last_hidden_prev.setZero();
            agent_ = AgentAffineVector{};
            agent_.resizeIfNeeded(hidden_size);
        }

        // serialization (weights are fixed; still useful for ckpt)
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
            const size_t wx = (size_t)W_x.size();
            const size_t wh = (size_t)W_h.size();
            const size_t vb = (size_t)b.size();
            // agent: gain + biasVec
            const size_t ag = 1 + (size_t)hidden_size;
            return wx + wh + vb + ag;
        }

    private:
        int input_size = 0;
        int hidden_size = 0;

        Eigen::MatrixXf W_x;
        Eigen::MatrixXf W_h;
        Eigen::VectorXf b;

        Eigen::VectorXf hidden_state;
        Eigen::VectorXf last_input;
        Eigen::VectorXf last_hidden_prev;

        AgentAffineVector agent_;

        static void initMatrix(Eigen::MatrixXf &M, float scale)
        {
            uint32_t seed = 1234567u;
            auto nextU = [&seed]()
            {
                seed = 1664525u * seed + 1013904223u;
                return (float)seed / (float)UINT32_MAX;
            };
            for (Eigen::Index r = 0; r < M.rows(); ++r)
                for (Eigen::Index c = 0; c < M.cols(); ++c)
                    M(r, c) = (2.0f * nextU() - 1.0f) * scale;
        }
    };
} // namespace ChromaFlow
