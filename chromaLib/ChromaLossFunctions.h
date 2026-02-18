#pragma once

#include "ChromaBaseClasses.h"

// TODO: Add documentation for each loss function

namespace ChromaFlow::LossFunctions
{

    
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

static inline FeatureTensor makeGrad(const Eigen::VectorXf& g)
{
    FeatureTensor out;
    out.data.resize(1, g.size());
    out.data.row(0) = g.transpose();
    out.numSamples = 1;
    out.features   = static_cast<int>(g.size());
    return out;
}


 
class ILoss
{
public:
    virtual FeatureTensor calculate(const FeatureTensor& output) const
    {
        Eigen::VectorXf y = getRowVector(output);
        return toFeatureTensor(Eigen::VectorXf::Zero(y.size()));
    }

    virtual FeatureTensor calculate(const FeatureTensor& input,
                                    const FeatureTensor& output) const
    {
        (void) input;
        return calculate(output);
    }

    virtual void setTarget(float t)
    {
        target = t;
    }

    float target = 0.0f;
};

class SelfSupervisedSibilanceLoss : public ILoss
{
public: 
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

// single input target losses
// setting the target value for the loss to a fixed value
// useful for normalizing the output of a network to a fixed range or task specific models(sometimes)

class RMSHomeostasisLoss : public ILoss
{
public: 
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

};
 

class CrestFactorLoss : public ILoss
{
public:

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
        Eigen::VectorXf grad = diff * y.normalized();
        return toFeatureTensor(grad);
    }
}; 

// interface for single input target losses
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
    void setAntiPhase(float maxAntiPhase)
    {
        this->maxAntiPhase = maxAntiPhase;
    }
};
} // namespace ChromaFlow
