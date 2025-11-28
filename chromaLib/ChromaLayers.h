 
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

class Collaborator : public DifferentiableModule
{
public:
    Collaborator (std::map<std::string, float> param_names_map, std::optional<std::unordered_set<std::string>> invert_names = std::nullopt)
        : invert_user (invert_names.value_or(std::unordered_set<std::string>{}))
    {
        param_names.reserve (param_names_map.size());
        for (const auto& kv : param_names_map)
        {
            param_names.push_back (kv.first);
            last_params_map[kv.first] = kv.second;
        }
    }

    FeatureTensor forward (const FeatureTensor& input) override { return input; }

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
            auto it = user_params.data.find(name);
            if (it != user_params.data.end())
            {
                intent = std::fmin(1.0f, std::fmax(0.0f, it->second));
            }

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
static float randUniform (float a, float b)
{
    static thread_local std::mt19937 rng([]{
        std::random_device rd;
        return std::mt19937(rd());
    }());
    std::uniform_real_distribution<float> dist(a, b);
    return dist(rng);
}

//
// --- convolutionalLayer (1D) ---
//
class convolutionalLayer : public DifferentiableModule
{
public:
    convolutionalLayer (int inputChannels, int outputChannels, int kernelSize)
    {
        (void) inputChannels; (void) outputChannels;
        kernelSize_ = std::max(1, kernelSize);
        kernel_.setZero(kernelSize_);
        biases_.setZero(1);
    }

    FeatureTensor forward (const FeatureTensor& input) override
    {
        last_input = input.data;
        const int K = kernelSize_;
        Eigen::VectorXf kernel = kernel_.size() == K ? kernel_ : Eigen::VectorXf::Ones(K);
        return forwardconvolve (input, kernel);
    }

    FeatureTensor forwardconvolve (const FeatureTensor& input, const Eigen::VectorXf& kernel)
    {
        const int K = static_cast<int>(kernel.size());
        if (K <= 0) return input;

        const int pad = K/2;
        const int inN = static_cast<int>(input.data.rows());
        const int outN = std::max(0, inN - K + 1) + 2*pad;

        FeatureTensor output;
        output.data = Eigen::MatrixXf::Zero(outN, 1);
        output.numSamples = outN;
        output.features = 1;

        Eigen::VectorXf col = input.data.col(0);
        const float* dataPtr = col.data();
        Eigen::VectorXf kr = kernel.reverse();
        float* outPtr = output.data.data();

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
            outPtr[i] = ChromaFlow::ChromaUtils::clip(v, 0.0f, 1.0f);
        }

        last_output = output.data;
        return output;
    }

    // Backprop: compute gradients wrt kernel and input; return pair<gradPrev, actPrev>
    std::pair<FeatureTensor, FeatureTensor> backward(const FeatureTensor& grad_out)
    {
        FeatureTensor grad_prev; // gradient to previous layer (same shape as last_input)
        FeatureTensor act_prev;  // activation of previous layer (last_input) as FeatureTensor

        if (grad_out.data.size() == 0) {
            grad_prev.data = Eigen::MatrixXf::Zero(last_input.rows(), last_input.cols());
            grad_prev.numSamples = last_input.rows();
            grad_prev.features = last_input.cols();
            act_prev.data = last_input;
            act_prev.numSamples = last_input.rows();
            act_prev.features = last_input.cols();
            return {grad_prev, act_prev};
        }

        Eigen::VectorXf g = grad_out.data.row(0).transpose(); // outN length
        const int K = kernelSize_;
        const int inN = static_cast<int>(last_input.rows());
        const int outN = static_cast<int>(g.size());
        const int pad = K/2;
        const int validOut = std::max(0, inN - K + 1);

        // grad kernel
        Eigen::VectorXf gradKernel = Eigen::VectorXf::Zero(K);
        Eigen::VectorXf inputCol = last_input.col(0);
        const float* dataPtr = inputCol.data();
        for (int i = 0; i < validOut; ++i)
        {
            float go = g[i + pad];
            if (go == 0.f) continue;
            Eigen::Map<const Eigen::VectorXf> segMap(dataPtr + i, K);
            gradKernel += go * segMap.reverse();
        }

        // grad input (convolve grad_out with kernel)
        Eigen::VectorXf gradInputVec = Eigen::VectorXf::Zero(inN);
        Eigen::VectorXf kr = kernel_.reverse();
        for (int i = 0; i < validOut; ++i)
        {
            float go = g[i + pad];
            if (go == 0.f) continue;
            for (int k = 0; k < K; ++k)
            {
                gradInputVec[i + k] += go * kr(k);
            }
        }

        // prepare FeatureTensor outputs
        grad_prev.data = Eigen::MatrixXf::Zero(inN, 1);
        grad_prev.data.col(0) = gradInputVec;
        grad_prev.numSamples = inN;
        grad_prev.features = 1;

        act_prev.data = last_input;
        act_prev.numSamples = last_input.rows();
        act_prev.features = last_input.cols();

        // optimizer update (use your optimizer)
        Eigen::VectorXf negGrad = -gradKernel; // negative as optimizer expects gradient sign convention
        optimizer.update(kernel_, negGrad, grad_prev);

        Eigen::VectorXf gradB(1);
        gradB(0) = g.sum();
        optimizer.update(biases_, -gradB, grad_prev);

        return {grad_prev, act_prev};
    }

    void learn (const FeatureTensor& gradient_from_output,
        const FeatureTensor& features_from_input,
        FeatureTensor& upstream_features) override
    {
        (void) features_from_input;
        (void) upstream_features;
        backward(gradient_from_output);
    }

    void setKernel(const Eigen::VectorXf& k) { kernel_ = k; kernelSize_ = static_cast<int>(k.size()); }
    void setBiases(const Eigen::VectorXf& b) { biases_ = b; }
    const Eigen::VectorXf& getWeights() const { return kernel_; }
    const Eigen::VectorXf& getBiases() const { return biases_; }

    void reset() override { last_input.setZero(); last_output.setZero(); }

    size_t getNumParams() const override {
        return static_cast<size_t>(kernel_.size()) + static_cast<size_t>(biases_.size());
    }

private:
    Eigen::VectorXf kernel_;
    Eigen::VectorXf biases_;
    int kernelSize_;
    Eigen::MatrixXf last_input;
    Eigen::VectorXf last_output;
    SGDWithMomentum optimizer;
};

//
// --- denseLayer ---
//
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
          optimizer_alpha (learningRate, momentum),
          gamma (Eigen::VectorXf::Ones(outputSize)),
          beta (Eigen::VectorXf::Zero(outputSize))
    {
        const float limit = std::sqrt (6.0f / (inputSize + outputSize));
        weights.resize (outputSize, inputSize);
        for (int o = 0; o < output_size; ++o)
            for (int i = 0; i < input_size; ++i)
                weights (o, i) = randUniform (-limit, limit);
        input_ema_state.setZero(input_size);
        learnable_alpha.setConstant(input_size, 0.5f);
    }

    FeatureTensor forward (const FeatureTensor& upstream_features) override
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

        for (int i=0;i<learnable_alpha.size();++i)
            learnable_alpha[i] = ChromaUtils::clip(learnable_alpha[i], 0.001f, 0.999f);

        Eigen::VectorXf one_minus_alpha = Eigen::VectorXf::Constant(input_size, 1.0f) - learnable_alpha;
        input_ema_state = input_ema_state.cwiseProduct(one_minus_alpha) + x.cwiseProduct(learnable_alpha);

        last_input_ema = input_ema_state;

        Eigen::VectorXf z = (weights * input_ema_state).eval();
        last_preact = z;

        if (use_layer_norm)
        {
            const float mean = z.mean();
            const float var = std::max (1e-6f, static_cast<float> ((z.array() - mean).square().mean()));
            Eigen::VectorXf norm = ((z.array() - mean) / std::sqrt (var)).matrix();
            last_xhat = norm;
            z = (gamma.cwiseProduct (norm) + beta).eval();
        }

        last_postnorm = z;

        Eigen::VectorXf a = z.unaryExpr ([this](float v) {
            switch (activation_type) {
                case ActivationType::LeakyRelu: return v >= 0.0f ? v : 0.01f * v;
                case ActivationType::Tanh: return std::tanh (v);
                case ActivationType::Sigmoid: return 1.0f / (1.0f + std::exp (-v));
                case ActivationType::Linear: default: return v;
            }
        }).eval();

        last_activation = a;

        FeatureTensor out;
        out.data.resize(1, static_cast<Eigen::Index>(a.size()));
        out.data.row(0) = a.transpose();
        out.numSamples = 1;
        out.features = static_cast<int>(a.size());
        return out;
    }

    std::pair<FeatureTensor, FeatureTensor> backward(const FeatureTensor& grad_output)
    {
        FeatureTensor grad_prev;
        FeatureTensor act_prev;

        if (grad_output.data.size() == 0) {
            grad_prev.data = Eigen::MatrixXf::Zero(last_input_ema.size(), 1);
            grad_prev.numSamples = static_cast<int>(last_input_ema.size());
            grad_prev.features = 1;
            act_prev.data = last_input_ema;
            act_prev.numSamples = static_cast<int>(last_input_ema.size());
            act_prev.features = 1;
            return {grad_prev, act_prev};
        }

        Eigen::VectorXf grad = grad_output.data.row(0).transpose(); // [output_size]

        // activation derivative
        Eigen::VectorXf dA(last_activation.size());
        for (int i = 0; i < last_activation.size(); ++i)
        {
            float v = last_postnorm(i);
            switch (activation_type) {
                case ActivationType::LeakyRelu: dA(i) = v >= 0.0f ? 1.0f : 0.01f; break;
                case ActivationType::Tanh: dA(i) = 1.0f - std::tanh(v) * std::tanh(v); break;
                case ActivationType::Sigmoid: {
                    float s = 1.0f / (1.0f + std::exp(-v));
                    dA(i) = s * (1.0f - s); break;
                }
                case ActivationType::Linear: default: dA(i) = 1.0f; break;
            }
        }

        Eigen::VectorXf dPre = grad.cwiseProduct(dA); // dL/d(preact after LN or raw preact)

        Eigen::VectorXf dgamma = Eigen::VectorXf::Zero(gamma.size());
        Eigen::VectorXf dbeta = dPre;
        Eigen::VectorXf dZ = dPre;

        if (use_layer_norm)
        {
            dgamma = dPre.cwiseProduct(last_xhat);
            const float mean = last_preact.mean();
            const float var = std::max(1e-6f, static_cast<float>((last_preact.array() - mean).square().mean()));
            const float invStd = 1.0f / std::sqrt(var);
            Eigen::VectorXf xmu = last_preact.array() - mean;
            Eigen::VectorXf dxhat = dPre.cwiseProduct(gamma);

            float sum_dxhat = dxhat.sum();
            float sum_dxhat_xmu = (dxhat.cwiseProduct(xmu)).sum();

            Eigen::VectorXf dz = (dxhat.array() - sum_dxhat / last_preact.size() - xmu.array() * (sum_dxhat_xmu / (var * last_preact.size()))).matrix() * invStd;
            dZ = dz;
        }

        // gradient of weights
        Eigen::MatrixXf dW = dZ * last_input_ema.transpose(); // [out x in]

        // gradient w.r.t input for previous layer
        Eigen::VectorXf dInput = weights.transpose() * dZ;

        grad_prev.data = Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(dInput.size()), 1);
        grad_prev.data.col(0) = dInput;
        grad_prev.numSamples = static_cast<int>(dInput.size());
        grad_prev.features = 1;

        act_prev.data = last_input_ema;
        act_prev.numSamples = static_cast<int>(last_input_ema.size());
        act_prev.features = 1;

        // updates
        optimizer.update(weights, -dW, grad_output);
        optimizer.update(gamma, -dgamma, grad_output);
        optimizer.update(beta, -dbeta, grad_output);

        // approximate alpha update if external input stored
        if (last_external_input.size() == learnable_alpha.size())
        {
            Eigen::VectorXf error_signal_mu = last_external_input - last_input_ema;
            Eigen::VectorXf d_alpha = (weights.transpose() * dZ).cwiseProduct(error_signal_mu).cwiseProduct(learnable_alpha.cwiseProduct(Eigen::VectorXf::Ones(learnable_alpha.size()) - learnable_alpha));
            optimizer_alpha.update(learnable_alpha, -d_alpha, grad_output);
        }

        return {grad_prev, act_prev};
    }

    void learn (const FeatureTensor& gradient_from_output,
        const FeatureTensor& features_from_input,
        FeatureTensor& upstream_features) override
    {
        (void) features_from_input;
        (void) upstream_features;
        backward(gradient_from_output);
    }

    const Eigen::MatrixXf& getWeights() const { return weights; }
    const Eigen::VectorXf& getGamma() const { return gamma; }
    const Eigen::VectorXf& getBeta() const { return beta; }
    const Eigen::VectorXf& getAlpha() const { return learnable_alpha; }

    void setWeights(const Eigen::MatrixXf& W) { weights = W; }
    void setGamma(const Eigen::VectorXf& G) { gamma = G; }
    void setBeta(const Eigen::VectorXf& B) { beta = B; }
    void setAlpha(const Eigen::VectorXf& A) { learnable_alpha = A; }

    void reset() override { input_ema_state.setZero(); last_input_ema.setZero(); last_preact.setZero(); last_postnorm.setZero(); last_activation.setZero(); last_xhat.setZero(); last_external_input.setZero(); }

    size_t getNumParams() const override {
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
    Eigen::VectorXf gamma; // Layer Norm Scale 
    Eigen::VectorXf beta;  // Layer Norm Shift (Fused Bias)

    Eigen::VectorXf input_ema_state; // Learnable Statistical Context (mu_t-1)
    Eigen::VectorXf learnable_alpha; // Trainable EMA decay rate (A_raw)

    // cached for backward
    Eigen::VectorXf last_input_ema;
    Eigen::VectorXf last_preact;
    Eigen::VectorXf last_postnorm;
    Eigen::VectorXf last_xhat;
    Eigen::VectorXf last_activation;
    Eigen::VectorXf last_external_input; // optional, if forward records raw external input

    SGDWithMomentum optimizer;
    SGDWithMomentum optimizer_alpha; 
};

//
// --- attentionLayer (conservative, stable backward) ---
//
class attentionLayer : public DifferentiableModule
{
public:
    attentionLayer (int inputSize,
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

        queryLayer = std::make_unique<ChromaFlow::denseLayer> (inputSize, outputSize, activationType, useLayerNorm, 0.0005f, 0.01f);
        keyLayer = std::make_unique<ChromaFlow::denseLayer> (inputSize, outputSize, activationType, useLayerNorm, 0.0005f, 0.01f);
        valueLayer = std::make_unique<ChromaFlow::denseLayer> (inputSize, outputSize, activationType, useLayerNorm, 0.0005f, 0.01f);
    }

    FeatureTensor forward (const FeatureTensor& input) override
    {
        last_input = input.data;

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
                // Fallback to uniform distribution to avoid zeroing output
                att = Eigen::VectorXf::Constant(expS.size(), 1.0f / std::max(1, (int)expS.size()));
            Eigen::VectorXf res = att.cwiseProduct(v);
            last_output_nonseq = res;
            FeatureTensor out;
            out.data.resize(1, static_cast<Eigen::Index>(res.size()));
            out.data.row(0) = res.transpose();
            out.numSamples = 1;
            out.features = static_cast<int>(res.size());
            return out;
        }

        const int seq_len = static_cast<int>(input.data.rows());
        const int d_in = static_cast<int>(input.data.cols());

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
                if (s > 0.0f) row /= s;
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

        last_output_vec = final_out;
        FeatureTensor outFt;
        outFt.data = final_out;
        outFt.numSamples = seq_len;
        outFt.features = d_model;
        return outFt;
    }

    // Backward: propagate gradient through Wo and approximate mapping to previous activation
    std::pair<FeatureTensor, FeatureTensor> backward(const FeatureTensor& grad_output)
    {
        FeatureTensor grad_prev;
        FeatureTensor act_prev;

        if (grad_output.data.size() == 0) {
            grad_prev.data = Eigen::MatrixXf::Zero(last_input.rows(), last_input.cols());
            grad_prev.numSamples = last_input.rows();
            grad_prev.features = last_input.cols();
            act_prev.data = last_input;
            act_prev.numSamples = last_input.rows();
            act_prev.features = last_input.cols();
            return {grad_prev, act_prev};
        }

        const int seq_len = static_cast<int>(last_input.rows());
        Eigen::MatrixXf dOutMat = grad_output.data * Wo; // [seq x d_model]

        // conservative mapping back to input: use average transpose of Wq/Wk/Wv
        Eigen::MatrixXf Wavg = (Wq.transpose() + Wk.transpose() + Wv.transpose()) / 3.0f; // [input x d_model]
        Eigen::MatrixXf gradInput = dOutMat * Wavg.transpose(); // [seq x input]

        grad_prev.data = gradInput;
        grad_prev.numSamples = static_cast<int>(gradInput.rows());
        grad_prev.features = static_cast<int>(gradInput.cols());

        act_prev.data = last_input;
        act_prev.numSamples = static_cast<int>(last_input.rows());
        act_prev.features = static_cast<int>(last_input.cols());

        // compute simple gradient for Wo: sum_t (grad_out_row outer output_mat_row)
        Eigen::MatrixXf gradWo = Eigen::MatrixXf::Zero(Wo.rows(), Wo.cols());
        for (int t = 0; t < seq_len; ++t)
        {
            Eigen::VectorXf gout = grad_output.data.row(t).transpose();
            Eigen::VectorXf outv = output_mat.row(t).transpose();
            gradWo += gout * outv.transpose();
        }
        if (training_allowed)
            Wo -= 1e-6f * gradWo;

        return {grad_prev, act_prev};
    }

    void learn (const FeatureTensor& gradient_from_output,
        const FeatureTensor& features_from_input,
        FeatureTensor& upstream_features) override
    {
        (void) features_from_input;
        (void) upstream_features;
        backward(gradient_from_output);
    }

    void reset() override
    {
        last_input.setZero();
        Q.setZero(); K.setZero(); V.setZero();
        output_mat.setZero(); final_out.setZero();
    }

    size_t getNumParams() const override {
        size_t total = static_cast<size_t>(Wq.size()) + static_cast<size_t>(Wk.size()) + static_cast<size_t>(Wv.size()) + static_cast<size_t>(Wo.size());
        if (queryLayer) total += queryLayer->getNumParams();
        if (keyLayer) total += keyLayer->getNumParams();
        if (valueLayer) total += valueLayer->getNumParams();
        return total;
    }

    const ChromaFlow::denseLayer* getQueryLayer() const { return queryLayer.get(); }
    const ChromaFlow::denseLayer* getKeyLayer() const { return keyLayer.get(); }
    const ChromaFlow::denseLayer* getValueLayer() const { return valueLayer.get(); }
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
    Eigen::VectorXf last_output_vec;
    std::vector<Eigen::MatrixXf> attn_weights;

    std::unique_ptr<ChromaFlow::denseLayer> queryLayer;
    std::unique_ptr<ChromaFlow::denseLayer> keyLayer;
    std::unique_ptr<ChromaFlow::denseLayer> valueLayer;
    bool training_allowed = false;

    static void initMatrix(Eigen::MatrixXf& M)
    {
        const float limit = std::sqrt(6.0f / (M.rows() + M.cols()));
        for (Eigen::Index r = 0; r < M.rows(); ++r)
            for (Eigen::Index c = 0; c < M.cols(); ++c)
                M(r,c) = randUniform(-limit, limit);
    }
};

//
// --- RNNCell with single-step backward (BPTT short) ---
//
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
        const float scale_x = std::sqrt (1.0f / static_cast<float> (input_size));
        const float scale_h = std::sqrt (1.0f / static_cast<float> (hidden_size));
        initMatrix (W_x, scale_x);
        initMatrix (W_h, scale_h);
        b.setZero();
    }

    FeatureTensor forward (const FeatureTensor& x_t) override
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
        last_output_vec = z.array().tanh().matrix();
        hidden_state = last_output_vec;

        out.data.resize(1, static_cast<Eigen::Index>(hidden_state.size()));
        out.data.row(0) = hidden_state.transpose();
        out.numSamples = 1;
        out.features = hidden_size;
        return out;
    }

    std::pair<FeatureTensor, FeatureTensor> backward(const FeatureTensor& grad_output)
    {
        FeatureTensor grad_prev, act_prev;
        if (grad_output.data.size() == 0) {
            grad_prev.data = Eigen::MatrixXf::Zero(last_input.size(), 1);
            grad_prev.numSamples = static_cast<int>(last_input.size());
            grad_prev.features = 1;
            act_prev.data = last_input;
            act_prev.numSamples = static_cast<int>(last_input.size());
            act_prev.features = 1;
            return {grad_prev, act_prev};
        }

        Eigen::VectorXf grad = grad_output.data.row(0).transpose(); // [H]
        Eigen::VectorXf dtanh = (1.0f - last_output_vec.array().square()).matrix();
        Eigen::VectorXf dPre = grad.cwiseProduct(dtanh); // [H]

        Eigen::MatrixXf grad_Wx = dPre * last_input.transpose(); // [H x I]
        Eigen::MatrixXf grad_Wh = dPre * last_hidden_prev.transpose(); // [H x H]
        Eigen::VectorXf grad_b = dPre;

        Eigen::VectorXf dInput = W_x.transpose() * dPre;

        optimizer.update(W_x, -grad_Wx, grad_output);
        optimizer.update(W_h, -grad_Wh, grad_output);
        optimizer.update(b, -grad_b, grad_output);

        grad_prev.data = Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(dInput.size()), 1);
        grad_prev.data.col(0) = dInput;
        grad_prev.numSamples = static_cast<int>(dInput.size());
        grad_prev.features = 1;

        act_prev.data = last_input;
        act_prev.numSamples = static_cast<int>(last_input.size());
        act_prev.features = 1;

        return {grad_prev, act_prev};
    }

    void learn (const FeatureTensor& gradient_from_output,
        const FeatureTensor& features_from_input,
        FeatureTensor& upstream_features) override
    {
        (void) features_from_input;
        (void) upstream_features;
        backward(gradient_from_output);
    }

    void reset() override
    {
        hidden_state.setZero();
        last_input.setZero();
        last_hidden_prev.setZero();
    }

    const Eigen::MatrixXf& getWx() const { return W_x; }
    const Eigen::MatrixXf& getWh() const { return W_h; }
    const Eigen::VectorXf& getB() const { return b; }
    const Eigen::VectorXf& getHiddenState() const { return hidden_state; }

    void setWx(const Eigen::MatrixXf& M) { W_x = M; }
    void setWh(const Eigen::MatrixXf& M) { W_h = M; }
    void setB(const Eigen::VectorXf& V) { b = V; }
    void setHiddenState(const Eigen::VectorXf& H) { hidden_state = H; }

    size_t getNumParams() const override {
        const size_t wx = static_cast<size_t>(W_x.rows()) * static_cast<size_t>(W_x.cols());
        const size_t wh = static_cast<size_t>(W_h.rows()) * static_cast<size_t>(W_h.cols());
        const size_t vb = static_cast<size_t>(b.size());
        return wx + wh + vb;
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

    Eigen::VectorXf last_preact;
    Eigen::VectorXf last_output_vec;

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
                M (h, i) = (2.0f * u - 1.0f) * scale;
            }
        }
    }
};

} // namespace ChromaFlow
