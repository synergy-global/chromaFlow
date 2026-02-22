/**
 * @file ChromaLossFunctions.h
 * @brief Loss functions for self-supervised and target-based training.
 */
#pragma once

#include "ChromaBaseClasses.h"

namespace ChromaFlow::LossFunctions
{
    /**
     * @brief Wrap a gradient vector as a single-row FeatureTensor.
     */
static inline FeatureTensor makeGrad(const Eigen::VectorXf& g)
{
    FeatureTensor out;
    out.data.resize(1, g.size());
    out.data.row(0) = g.transpose();
    out.numSamples = 1;
    out.features   = static_cast<int>(g.size());
    return out;
}


 
/**
 * @brief Base class for differentiable loss functions.
 *
 * Provides default no-op implementations so subclasses
 * can override only the signatures they need.
 */
class ILoss
{
public:
    /**
     * @brief Calculate gradient with respect to a single output tensor.
     *
     * Default implementation returns a zero gradient.
     */
    virtual FeatureTensor calculate(const FeatureTensor& output) const
    {
        Eigen::VectorXf y = getRowVector(output);
        return toFeatureTensor(Eigen::VectorXf::Zero(y.size()));
    }



    /**
     * @brief Calculate gradient given input and output tensors.
     *
     * Default implementation forwards to the single-tensor overload.
     */
    virtual FeatureTensor calculate(const FeatureTensor& input,
                                    const FeatureTensor& output) const
    {
        (void) input;
        return calculate(output);
    }



    /**
     * @brief Set a scalar target used by the loss.
     */
    virtual void setTarget(float t)
    {
        target = t;
    }

    float target = 0.0f;
};

/**
 * @brief Two-input loss for controlling sibilance while preserving energy.
 */
class SelfSupervisedSibilanceLoss : public ILoss
{
public:
    /**
     * @brief Compute gradient using high-frequency and RMS features.
     */
    FeatureTensor calculate(const FeatureTensor& input,
                            const FeatureTensor& output) const override
    {
        const int N = output.data.cols();
        if (N < 2)
            return zeroGrad(N);

        Eigen::VectorXf in  = input.data.row(0).transpose();
        Eigen::VectorXf out = output.data.row(0).transpose();

        float inHF   = in(0);
        float inRMS  = in(1);

        float outHF  = out(0);
        float outRMS = out(1);

        // --- 1️⃣ High-frequency reduction term ---
        // Want HF reduced but not zero
        float hfRatio = outHF / (inHF + 1e-6f);
        float hfTarget = 0.6f; // ideal reduction ratio
        float hfLossGrad = 2.0f * (hfRatio - hfTarget);

        // --- 2️⃣ Energy preservation term ---
        float rmsDiff = outRMS - inRMS;
        float rmsLossGrad = 2.0f * rmsDiff;

        // --- Combine ---
        Eigen::VectorXf grad = Eigen::VectorXf::Zero(N);

        // Gradient flows back through output features
        grad(0) = hfLossGrad;
        grad(1) = rmsLossGrad;

        FeatureTensor outGrad;
        outGrad.data.resize(1, N);
        outGrad.data.row(0) = grad.transpose();
        outGrad.numSamples = 1;
        outGrad.features = N;

        return outGrad;
    }

private:
    FeatureTensor zeroGrad(int N) const
    {
        FeatureTensor g;
        g.data = Eigen::MatrixXf::Zero(1, N);
        g.numSamples = 1;
        g.features = N;
        return g;
    }
}; 

/**
 * @brief RMS homeostasis loss with scalar target.
 *
 * Drives RMS of the output toward @c target.
 */
class RMSHomeostasisLoss : public ILoss
{
public: 
    /// Numerical stability epsilon.
    float eps = 1e-8f;

    FeatureTensor calculate(const FeatureTensor& output) const override
    {
        Eigen::VectorXf y = getRowVector(output);
        const int N = (int)y.size();
        if (N == 0)
            return toFeatureTensor(y);

        float rms = std::sqrt(y.array().square().mean() + eps);
        float diff = rms - target;

        Eigen::VectorXf grad = (2.0f / (float)N) * diff * (y / (rms + eps));
        return toFeatureTensor(grad);
    } 
    /**
     * @brief Set RMS target value.
     */
    void setTarget(float t) override
    {
        target = t;
    }
};
 

/**
 * @brief Crest factor loss (ratio of peak to RMS).
 */
class CrestFactorLoss : public ILoss
{
public:
    /**
     * @brief Set desired crest factor target.
     */
    void setTarget(float t) override
    {
        target = t;
    }

    /// Numerical stability epsilon.
    float eps = 1e-8f;

    FeatureTensor calculate(const FeatureTensor& output) const override
    {
        Eigen::VectorXf y = getRowVector(output);
        const int N = (int)y.size();
        if (N == 0)
            return toFeatureTensor(y);

        float peak = y.cwiseAbs().maxCoeff();
        float rms  = std::sqrt(y.array().square().mean() + eps);
        float crest = peak / (rms + eps);

        float diff = crest - target;

        // approximate gradient: push toward RMS scaling
        Eigen::VectorXf grad = diff * y / (rms + eps);
        return toFeatureTensor(grad);
    }
}; 

/**
 * @brief Spectral tilt loss between low and high bands.
 */
class SpectralTiltLoss : public ILoss
{
public:

    float eps = 1e-8f;

    FeatureTensor calculate(const FeatureTensor& output) const override
    {
        Eigen::VectorXf f = getRowVector(output);
        if (f.size() < 2)
            return toFeatureTensor(f);

        float low  = std::max(f(0), eps);
        float high = std::max(f(1), eps);

        float tilt = std::log(high) - std::log(low);
        float diff = tilt - target;

        Eigen::VectorXf grad = Eigen::VectorXf::Zero(f.size());
        grad(0) = -diff / low;
        grad(1) =  diff / high;

        return toFeatureTensor(grad);
    }
};
 
/**
 * @brief Spectral flux loss between consecutive frames.
 *
 * Uses an internal previous-frame state.
 */
class SpectralFluxLoss : public ILoss
{
public:

    mutable Eigen::VectorXf prev;
    mutable bool hasPrev = false;

    FeatureTensor calculate(const FeatureTensor& output) const override
    {
        Eigen::VectorXf y = getRowVector(output);

        if (!hasPrev)
        {
            prev = y;
            hasPrev = true;
            return toFeatureTensor(Eigen::VectorXf::Zero(y.size()));
        }

        Eigen::VectorXf diff = y - prev;
        prev = target * prev + (1.0f - target) * y;

        return toFeatureTensor(diff);
    }
};

/**
 * @brief Two-input energy conservation loss.
 *
 * Penalises mismatch between input and output energy.
 */
class EnergyConservationLoss : public ILoss
{
public:
    float eps = 1e-8f;

    FeatureTensor calculate(const FeatureTensor& input,
                            const FeatureTensor& output) const override
    {
        Eigen::VectorXf y = getRowVector(output);
        Eigen::VectorXf x = getRowVector(input);

        if (y.size() == 0 || x.size() == 0)
            return toFeatureTensor(y);

        float Ey = y.array().square().mean();
        float Ex = x.array().square().mean();

        float diff = Ey - Ex;

        Eigen::VectorXf grad = (2.0f / (float)y.size()) * diff * y;
        return toFeatureTensor(grad);
    }
};
 

/**
 * @brief Loss that penalises excess harmonic energy above a target.
 */
class HarmonicEnergyLoss : public ILoss
{
public:
    FeatureTensor calculate(const FeatureTensor& output) const override
    {
        Eigen::VectorXf y = getRowVector(output);
        if (y.size() == 0)
            return toFeatureTensor(y);

        float energy = y.array().square().mean();
        float diff = std::max(0.0f, energy - target);

        Eigen::VectorXf grad = 2.0f * diff * y;
        return toFeatureTensor(grad);
    }
};

/**
 * @brief Neural compressor loss function.
 *
 * Penalises RMS reduction, crest factor, and transient energy.
 */
class NeuralCompressorLoss : public ILoss
{
public:
    FeatureTensor calculate(const FeatureTensor& input,
                            const FeatureTensor& output) const override
    {
        Eigen::VectorXf x = getRowVector(input);
        Eigen::VectorXf y = getRowVector(output);

        const int N = (int)y.size();
        if (N == 0)
            return toFeatureTensor(y);

        // =========================
        // 1️⃣ RMS CONTROL
        // =========================
        float rmsIn  = std::sqrt(x.array().square().mean() + eps);
        float rmsOut = std::sqrt(y.array().square().mean() + eps);

        float targetRms = rmsIn * (1.0f - rmsReduction);
        float rmsDiff   = rmsOut - targetRms;

        Eigen::VectorXf gradRms =
            (2.0f / N) * rmsDiff * (y / (rmsOut + eps));

        // =========================
        // 2️⃣ CREST FACTOR CONTROL
        // =========================
        float peak  = y.cwiseAbs().maxCoeff();
        float crest = peak / (rmsOut + eps);

        float crestDiff = crest - crestTarget;

        Eigen::VectorXf gradCrest =
            crestDiff * (y / (rmsOut + eps));

        // =========================
        // 3️⃣ TRANSIENT CONTROL
        // =========================
        Eigen::VectorXf diffVec = y - prev;
        float transientEnergy = diffVec.array().abs().mean();

        float tDiff = transientEnergy - transientTarget;

        Eigen::VectorXf gradTransient =
            (tDiff > 0.0f ? tDiff : 0.0f) * diffVec;

        // update previous frame (EMA)
        prev = 0.95f * prev + 0.05f * y;

        // =========================
        // Combine (bounded)
        // =========================
        Eigen::VectorXf grad =
              wRms  * gradRms
            + wCrest * gradCrest
            + wTransient * gradTransient;

        // Soft clip gradient (critical for RT safety)
        grad = grad.unaryExpr([](float v)
        {
            const float limit = 5.0f;
            return std::max(-limit, std::min(limit, v));
        });

        return toFeatureTensor(grad);
    }
// target setters
    void setTarget(float rmsReduction, float crestTarget, float transientTarget)
    {
        this->rmsReduction = rmsReduction;
        this->crestTarget  = crestTarget;
        this->transientTarget = transientTarget;
    }
private:

    // --- User Targets ---
    float rmsReduction = 0.15f;      // 15% RMS reduction
    float crestTarget  = 3.0f;       // desired crest factor
    float transientTarget = 0.02f;   // transient energy cap

    // --- Weights ---
    float wRms  = 0.4f;
    float wCrest = 0.4f;
    float wTransient = 0.2f;

    float eps = 1e-8f;

    mutable Eigen::VectorXf prev = Eigen::VectorXf::Zero(1);
};
// ============================================================
// 1️⃣ Stereo Width Target Loss
// Encourages M/S energy ratio toward target width
// ============================================================

class StereoWidthLoss : public ILoss
{
public:
    float eps = 1e-8f;

    FeatureTensor calculate(const FeatureTensor& Output) const override
    {
        Eigen::VectorXf y = getRowVector(Output);
        if (y.size() < 2)
            return toFeatureTensor(y);

        float L = y(0);
        float R = y(1);

        float mid  = 0.5f * (L + R);
        float side = 0.5f * (L - R);

        float midE  = mid * mid + eps;
        float sideE = side * side + eps;

        float width = std::sqrt(sideE / midE);
        float diff = width - target;

        Eigen::VectorXf grad(2);

        // Push side relative to mid
        grad(0) =  diff * (L - R);
        grad(1) = -diff * (L - R);

        return toFeatureTensor(grad);
    }
};

// ============================================================
// 2️⃣ Correlation Control Loss
// Encourages correlation toward target
// ============================================================

class StereoCorrelationLoss : public ILoss
{
public:
    float eps = 1e-8f;

    FeatureTensor calculate(const FeatureTensor& Output) const override
    {
        Eigen::VectorXf y = getRowVector(Output);
        if (y.size() < 2)
            return toFeatureTensor(y);

        float L = y(0);
        float R = y(1);

        float corr = (L * R) / (std::sqrt(L*L * R*R) + eps);
        float diff = corr - target;

        Eigen::VectorXf grad(2);
        grad(0) = diff * R;
        grad(1) = diff * L;

        return toFeatureTensor(grad);
    }
};

// ============================================================
// 3️⃣ Mono Compatibility Loss
// Penalises destructive phase cancellation
// ============================================================

class MonoCompatibilityLoss : public ILoss
{
public:
    float eps = 1e-8f;

    FeatureTensor calculate(const FeatureTensor& Output) const override
    {
        Eigen::VectorXf y = getRowVector(Output);
        if (y.size() < 2)
            return toFeatureTensor(y);

        float L = y(0);
        float R = y(1);

        float mono = 0.5f * (L + R);
        float stereoE = L*L + R*R;
        float monoE   = mono * mono * 2.0f;

        float diff = stereoE - monoE;

        Eigen::VectorXf grad(2);
        grad(0) = diff * (L - mono);
        grad(1) = diff * (R - mono);

        return toFeatureTensor(grad);
    }
};

// ============================================================
// 4️⃣ Spatial Energy Balance Loss
// Prevents one side dominance
// ============================================================

class StereoBalanceLoss : public ILoss
{
public:
    float eps = 1e-8f;

    FeatureTensor calculate(const FeatureTensor& Output) const override
    {
        Eigen::VectorXf y = getRowVector(Output);
        if (y.size() < 2)
            return toFeatureTensor(y);

        float L = y(0);
        float R = y(1);

        float diff = (L*L) - (R*R);

        Eigen::VectorXf grad(2);
        grad(0) = 2.0f * diff * L;
        grad(1) = -2.0f * diff * R;

        return toFeatureTensor(grad);
    }
};

// ============================================================
// 5️⃣ Decorrelated Width Expansion Loss
// Encourages side energy but penalises extreme anti-phase
// ============================================================

class ControlledExpansionLoss : public ILoss
{
public:
    float maxAntiPhase = -0.7f;
    float eps = 1e-8f;

    FeatureTensor calculate(const FeatureTensor& Output) const override
    {
        Eigen::VectorXf y = getRowVector(Output);
        if (y.size() < 2)
            return toFeatureTensor(y);

        float L = y(0);
        float R = y(1);

        float mid  = 0.5f * (L + R);
        float side = 0.5f * (L - R);

        float sideE = side * side;
        float corr  = (L * R) / (std::sqrt(L*L * R*R) + eps);

        float widthDiff = sideE - target;
        float corrPenalty = std::min(0.0f, corr - maxAntiPhase);

        Eigen::VectorXf grad(2);

        grad(0) = widthDiff * (L - R) + corrPenalty * R;
        grad(1) = -widthDiff * (L - R) + corrPenalty * L;

        return toFeatureTensor(grad);
    }

    /// Set maximum allowed anti-phase correlation.
    void setAntiPhase(float maxAntiPhase)
    {
        this->maxAntiPhase = maxAntiPhase;
    }
};
} // namespace ChromaFlow
