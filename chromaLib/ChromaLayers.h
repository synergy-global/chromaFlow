/**
 * @file ChromaLayers.h
 * @brief Real-time neural DSP layers and supporting utilities.
 */
#pragma once
#include "./ChromaBaseClasses.h"

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

/**
 * @brief Trainable layers built on top of ChromaFlow primitives.
 */
namespace ChromaFlow::Layers
{

    /**
     * @brief Simple 1D convolutional layer with IIR-style learning.
     */
    class convolutionalLayer : public DifferentiableModule
    {
    public:
        explicit convolutionalLayer(int kernelSize)
            : K(kernelSize)
        {
            weights.resize(K);
            grad.resize(K);
            buffer.resize(K);
            buffer.setZero();
            grad.setZero();

            for (int i = 0; i < K; ++i)
                weights(i) = randUniform(-0.01f, 0.01f);
        }

        /// Set kernel coefficients from a vector (truncated if necessary).
        void setKernel(const Eigen::VectorXf &w)
        {
            const int n = std::min<int>(K, w.size());
            for (int i = 0; i < n; ++i)
                weights(i) = w(i);
        }

        /// Set bias term applied after convolution.
        void setBias(float b)
        {
            bias = b;
        }

        /**
         * @brief Forward pass: process a single-sample FeatureTensor.
         */
        FeatureTensor forward(const ChromaFlow::FeatureTensor &x) override
        {
            // shift ring buffer
            for (int i = K - 1; i > 0; --i)
                buffer(i) = buffer(i - 1);
            buffer(0) = x.data(0, 0);

            float y = weights.dot(buffer) + bias;
            last_output = y;
            ChromaFlow::FeatureTensor out;
            out.data.resize(1, 1);
            out.data(0, 0) = y;
            out.numSamples = 1;
            out.features = 1;
            return out;
        }

        /**
         * @brief Learn from a scalar error using truncated IIR backprop.
         */
        void learn(const ChromaFlow::FeatureTensor &error) override
        {
            if (error.features != 1)
                return;

            // IIR truncated BPTT
            delta = deltaAlpha * delta + (1.0f - deltaAlpha) * errorVectorRow0(error)(0);

            // instant gradient
            Eigen::VectorXf dW = delta * buffer;

            // smooth gradient
            grad = gradAlpha * grad + (1.0f - gradAlpha) * dW;

            // update
            weights -= learningRate * grad;

            // clip
            for (int i = 0; i < K; ++i)
                weights(i) = clipf(weights(i), -wMax, wMax);
        }

    private:
        int K;
        Eigen::VectorXf weights;
        Eigen::VectorXf grad;
        Eigen::VectorXf buffer;

        float last_output = 0.0f;
        float delta = 0.0f;
        float bias = 0.0f;

        float learningRate = 1e-6f;
        float deltaAlpha = 0.98f;
        float gradAlpha = 0.995f;
        float wMax = 2.0f;
    };

    /**
     * @brief Fully-connected layer with IIR-style gradient accumulation.
     */
    class DenseLayer : public DifferentiableModule
    {
    public:
        DenseLayer(int inSize, int outSize)
            : input_size(inSize),
              output_size(outSize)
        {
            const float limit = std::sqrt(6.0f / (inSize + outSize));

            weights.resize(outSize, inSize);
            gradW.resize(outSize, inSize);
            eligibility.resize(outSize, inSize); // 🔥 FIX
            weights.setZero();
            gradW.setZero();
            eligibility.setZero(); // 🔥 FIX

            for (int r = 0; r < outSize; ++r)
                for (int c = 0; c < inSize; ++c)
                    weights(r, c) = randUniform(-limit, limit);

            last_input.resize(inSize);
            last_output.resize(outSize);

            delta.resize(outSize);
            delta.setZero();
        }

        FeatureTensor forward(const ChromaFlow::FeatureTensor &in) override
        {
            Eigen::VectorXf x = Eigen::VectorXf::Zero(input_size);

            if (in.data.rows() == 1)
            {
                int n = std::min((int)in.data.cols(), input_size);
                x.head(n) = in.data.row(0).transpose().head(n);
            }

            last_input = x;

            Eigen::VectorXf y = weights * x;
            last_output = y;

            ChromaFlow::FeatureTensor out;
            out.data.resize(1, output_size);
            out.data.row(0) = y.transpose();
            out.numSamples = 1;
            out.features = output_size;
            return out;
        }

        // --- IIR BACKPROP ---
        void learn(const ChromaFlow::FeatureTensor &error) override
        {
            if (error.features != output_size)
                return;

            Eigen::VectorXf e = errorVectorRow0(error);

            if (e.size() != output_size)
            {
                Eigen::VectorXf tmp = Eigen::VectorXf::Zero(output_size);
                int n = std::min((int)e.size(), output_size);
                tmp.head(n) = e.head(n);
                e = tmp;
            }

            //  IIR delta accumulation (truncated BPTT approx)
            delta = deltaAlpha * delta + (1.0f - deltaAlpha) * e;

            //  Instantaneous gradient
            Eigen::MatrixXf dW = delta * last_input.transpose();

            //  IIR gradient smoothing
            gradW = gradAlpha * gradW + (1.0f - gradAlpha) * dW;

            // IIR eligibility update
            eligibility = alpha * eligibility + gradW;
            // Clip eligibility
            eligibility = eligibility.unaryExpr([&](float v)
                                                { return std::clamp(v, -gradClip, gradClip); });

            // Parameter update
            weights -= lr * eligibility;

            //  Hard clip weights (critical for RT stability)
            weights = weights.unaryExpr([this](float v)
                                        { return clipf(v, -wMax, wMax); });
        }

        void reset() override
        {
            delta.setZero();
            gradW.setZero();
        }

        size_t getNumParams() const override
        {
            return (size_t)weights.size();
        }

    private:
        int input_size = 0;
        int output_size = 0;

        Eigen::MatrixXf weights;
        Eigen::MatrixXf gradW;

        Eigen::VectorXf last_input;
        Eigen::VectorXf last_output;

        Eigen::VectorXf delta; // IIR error memory

        // --- Stability knobs ---
        float learningRate = 1e-5f;
        float deltaAlpha = 0.98f;    // BPTT truncation memory
        float gradAlpha = 0.995f;    // gradient smoothing
        float wMax = 3.0f;           // hard bound
        Eigen::MatrixXf eligibility; // same size as weights
        float alpha = 0.9f;
        float lr = 0.001f;
        float gradClip = 1.0f;
    };
    class attentionLayer : public DifferentiableModule
    {
    public:
        attentionLayer(int inputSize,
                       int numHeads,
                       int outputSize)
            : heads(numHeads),
              d_model(outputSize),
              d_k(std::max(1, outputSize / std::max(1, numHeads)))
        {
            jassert(outputSize % numHeads == 0);

            queryLayer = std::make_unique<DenseLayer>(inputSize, outputSize);
            keyLayer = std::make_unique<DenseLayer>(inputSize, outputSize);
            valueLayer = std::make_unique<DenseLayer>(inputSize, outputSize);
        }

        ChromaFlow::FeatureTensor forward(const ChromaFlow::FeatureTensor &input) override
        {
            last_input = input.data;

            // SINGLE VECTOR path (your use case)
            if (input.data.rows() <= 1)
            {
                ChromaFlow::FeatureTensor qT = queryLayer->forward(input);
                ChromaFlow::FeatureTensor kT = keyLayer->forward(input);
                ChromaFlow::FeatureTensor vT = valueLayer->forward(input);

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

                ChromaFlow::FeatureTensor out;
                out.data.resize(1, (Eigen::Index)res.size());
                out.data.row(0) = res.transpose();
                out.numSamples = 1;
                out.features = (int)res.size();
                last_out_vec = res;
                return out;
            }

            const int seq_len = (int)input.data.rows();
            ChromaFlow::FeatureTensor out = input; // fallback
            out.data = Eigen::MatrixXf::Zero(seq_len, d_model);
            out.numSamples = seq_len;
            out.features = d_model;

            // Very conservative: just apply per-row dense Q/K/V + per-row elementwise attention
            for (int t = 0; t < seq_len; ++t)
            {
                ChromaFlow::FeatureTensor step;
                step.data.resize(1, input.data.cols());
                step.data.row(0) = input.data.row(t);
                step.numSamples = 1;
                step.features = (int)input.data.cols();

                ChromaFlow::FeatureTensor qT = queryLayer->forward(step);
                ChromaFlow::FeatureTensor kT = keyLayer->forward(step);
                ChromaFlow::FeatureTensor vT = valueLayer->forward(step);

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

                out.data.row(t) = res.transpose();
            }

            return out;
        }

        void learn(const ChromaFlow::FeatureTensor &error) override
        {
            queryLayer->learn(error);
            keyLayer->learn(error);
            valueLayer->learn(error);
        }

        void reset() override
        {
            last_input.setZero();
            last_out_vec.setZero();
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

        const DenseLayer *getQueryLayer() const { return queryLayer.get(); }
        const DenseLayer *getKeyLayer() const { return keyLayer.get(); }
        const DenseLayer *getValueLayer() const { return valueLayer.get(); }

    private:
        int heads = 4;
        int d_model = 0;
        int d_k = 0;

        Eigen::MatrixXf last_input;
        Eigen::VectorXf last_out_vec;

        std::unique_ptr<DenseLayer> queryLayer;
        std::unique_ptr<DenseLayer> keyLayer;
        std::unique_ptr<DenseLayer> valueLayer;
    };

    class RNNCell : public DifferentiableModule
    {
    public:
        RNNCell(int inputSize, int hiddenSize)
            : I(inputSize), H(hiddenSize)
        {
            Wx = Eigen::MatrixXf::Random(H, I) * 0.1f;
            Wh = Eigen::MatrixXf::Random(H, H) * 0.1f;

            gradWx = Eigen::MatrixXf::Zero(H, I);
            gradWh = Eigen::MatrixXf::Zero(H, H);

            h = Eigen::VectorXf::Zero(H);
            delta = Eigen::VectorXf::Zero(H);
        }

        FeatureTensor forward(const ChromaFlow::FeatureTensor &x) override
        {
            last_x = x.data.row(0).transpose();
            last_h = h;

            Eigen::VectorXf pre = Wx * last_x + Wh * last_h;
            h = pre.array().tanh().matrix();

            ChromaFlow::FeatureTensor out;
            out.data.resize(1, H);
            out.data.row(0) = h.transpose();
            out.numSamples = 1;
            out.features = H;

            return out;
        }

        void learn(const ChromaFlow::FeatureTensor &error) override
        {
            if (error.features != H)
                return;

            // IIR truncated credit assignment
            delta = deltaAlpha * delta + (1.0f - deltaAlpha) * error.data.row(0).transpose();

            // derivative of tanh
            Eigen::VectorXf dtanh = (1.0f - h.array().square()).matrix();
            Eigen::VectorXf local = delta.cwiseProduct(dtanh);

            // gradients
            Eigen::MatrixXf dWx = local * last_x.transpose();
            Eigen::MatrixXf dWh = local * last_h.transpose();

            // IIR smoothing
            gradWx = gradAlpha * gradWx + (1.0f - gradAlpha) * dWx;
            gradWh = gradAlpha * gradWh + (1.0f - gradAlpha) * dWh;

            // update
            Wx -= learningRate * gradWx;
            Wh -= learningRate * gradWh;

            // stability clamp
            Wx = Wx.unaryExpr([this](float v)
                              { return clipf(v, -wMax, wMax); });

            Wh = Wh.unaryExpr([this](float v)
                              { return clipf(v, -wMax, wMax); });
        }

    private:
        int I, H;

        Eigen::MatrixXf Wx, Wh;
        Eigen::MatrixXf gradWx, gradWh;

        Eigen::VectorXf h;
        Eigen::VectorXf last_h;
        Eigen::VectorXf last_x;

        Eigen::VectorXf delta;

        float learningRate = 5e-6f;
        float deltaAlpha = 0.97f;
        float gradAlpha = 0.995f;
        float wMax = 2.0f;
    };

    // layer normalization
    class LayerNorm : public DifferentiableModule
    {
    public:
        LayerNorm(int features)
            : d_model(features)
        {
            gain = Eigen::VectorXf::Ones(d_model);
            biasVec = Eigen::VectorXf::Zero(d_model);
        }

        ChromaFlow::FeatureTensor forward(const ChromaFlow::FeatureTensor &input) override
        {
            ChromaFlow::FeatureTensor out = input;
            const int rows = static_cast<int>(input.data.rows());
            const int cols = static_cast<int>(input.data.cols());
            if (rows > 0 && cols > 0)
            {
                const float eps = 1e-6f;
                Eigen::VectorXf mean = input.data.rowwise().mean();
                out.data = input.data.colwise() - mean;
                Eigen::VectorXf var = out.data.array().square().rowwise().mean();
                Eigen::VectorXf denom = (var.array() + eps).sqrt().matrix();
                out.data = out.data.array().colwise() / denom.array();
                out.data = (out.data.array().rowwise() * gain.transpose().array()).matrix();
                out.data = (out.data.array().rowwise() + biasVec.transpose().array()).matrix();
            }
            return out;
        }

    private:
        int d_model;
        Eigen::VectorXf gain;
        Eigen::VectorXf biasVec;
    };

    // max pooling
    class MaxPool : public DifferentiableModule
    {
    public:
        MaxPool(int poolSize)
            : poolSize(poolSize)
        {
        }

        ChromaFlow::FeatureTensor forward(const ChromaFlow::FeatureTensor &input) override
        {
            ChromaFlow::FeatureTensor out = input;
            out.data = input.data.block(0, 0, input.data.rows(), input.data.cols() - poolSize + 1)
                           .colwise()
                           .maxCoeff()
                           .transpose();
            out.numSamples = 1;
            out.features = (int)out.data.size();
            return out;
        }

    private:
        int poolSize;
    };
} // namespace ChromaFlow
